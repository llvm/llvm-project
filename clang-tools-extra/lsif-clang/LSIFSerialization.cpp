//===-- LSIFSerialization.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// LSIF spec:
// https://microsoft.github.io/language-server-protocol/specifications/lsif/0.5.0/specification/
//
//===----------------------------------------------------------------------===//

#include "LSIFSerialization.h"
#include "Index.h"
#include "Relation.h"
#include "Serialization.h"
#include "SymbolLocation.h"
#include "SymbolOrigin.h"
#include "dex/Dex.h"
#include "index/Symbol.h"
#include "index/SymbolID.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>

namespace llvm {
template <> struct DenseMapInfo<clang::clangd::SymbolLocation> {
  using Entry = clang::clangd::SymbolLocation;
  static inline Entry getEmptyKey() {
    static Entry E{};
    return E;
  }
  static inline Entry getTombstoneKey() {
    static Entry E{{}, {}, "TOMBSTONE"};
    return E;
  }
  static unsigned getHashValue(const Entry &Val) {
    return llvm::hash_combine(Val.Start.line(), Val.Start.column(),
                              Val.End.line(), Val.End.column(),
                              llvm::StringRef(Val.FileURI));
  }
  static bool isEqual(const Entry &LHS, const Entry &RHS) {
    return llvm::StringRef(LHS.FileURI) == llvm::StringRef(RHS.FileURI) &&
           LHS.Start == RHS.Start && LHS.End == RHS.End;
  }
};
} // namespace llvm

namespace {
struct LSIFMeta {
  LSIFMeta(llvm::raw_ostream &OS, const clang::clangd::IndexFileOut &O)
    : OS(OS), ProjectRoot(O.ProjectRoot), Debug(O.Debug), DebugFiles(O.DebugFiles) {}
  llvm::raw_ostream &OS;
  const std::string ProjectRoot;
  const bool Debug;
  const bool DebugFiles;
  int IDCounter = 0;
  llvm::StringMap<int> DocumentIDs;
  llvm::DenseMap<clang::clangd::SymbolID, int> ReferenceResultIDs;
  llvm::DenseMap<clang::clangd::SymbolID, int> ResultSetIDs;
  llvm::DenseMap<clang::clangd::SymbolLocation, int> RangeIDs;

  bool contains(const char *FileURI) {
    bool contains = FileURI && llvm::StringRef(FileURI).startswith(ProjectRoot);
    if (!contains && DebugFiles) {
      llvm::errs() << "Ignoring file: " << llvm::StringRef(FileURI) << "\n";
    }
    return contains;
  }
};

using JOStream = llvm::json::OStream;

int writeVertex(LSIFMeta &Meta, const std::string &Label,
                llvm::function_ref<void(JOStream &)> Contents) {
  JOStream JSONOut(Meta.OS);
  JSONOut.object([&] {
    JSONOut.attribute("id", Meta.IDCounter);
    JSONOut.attribute("type", "vertex");
    JSONOut.attribute("label", Label);
    Contents(JSONOut);
  });
  Meta.OS << "\n";
  return Meta.IDCounter++;
}

void writeEdge(LSIFMeta &Meta, const std::string &Label, int OutV,
               llvm::function_ref<void(JOStream &)> Contents) {
  JOStream JSONOut(Meta.OS);
  JSONOut.object([&] {
    JSONOut.attribute("id", Meta.IDCounter++);
    JSONOut.attribute("type", "edge");
    JSONOut.attribute("label", Label);
    JSONOut.attribute("outV", OutV);
    Contents(JSONOut);
  });
  Meta.OS << "\n";
}

void write1to1Edge(LSIFMeta &Meta, const std::string &Label, int OutV,
                   int InV) {
  writeEdge(Meta, Label, OutV,
            [&](JOStream &JSONOut) { JSONOut.attribute("inV", InV); });
}

void write1toNEdge(LSIFMeta &Meta, const std::string &Label, int OutV,
                   const std::vector<int> &InVs,
                   llvm::function_ref<void(JOStream &)> Contents) {
  writeEdge(Meta, Label, OutV, [&](JOStream &JSONOut) {
    JSONOut.attributeArray("inVs", [&] {
      for (auto InV : InVs) {
        JSONOut.value(InV);
      }
    });
    Contents(JSONOut);
  });
}

void writeContainsEdge(LSIFMeta &Meta, int OutV, const std::vector<int> &InVs) {
  write1toNEdge(Meta, "contains", OutV, InVs, [&](JOStream &JSONOut) {});
}

void writeItemEdge(LSIFMeta &Meta, int OutV, const std::vector<int> &InVs,
                   int DocumentID) {
  write1toNEdge(Meta, "item", OutV, InVs, [&](JOStream &JSONOut) {
    JSONOut.attribute("document", DocumentID);
  });
}

void writeItemEdge(LSIFMeta &Meta, int OutV, const std::vector<int> &InVs,
                   int DocumentID, const std::string &Type) {
  write1toNEdge(Meta, "item", OutV, InVs, [&](JOStream &JSONOut) {
    JSONOut.attribute("document", DocumentID);
    JSONOut.attribute("property", Type);
  });
}

void writePositionField(
    JOStream &JSONOut, const std::string &FieldName,
    const clang::clangd::SymbolLocation::Position &Position) {
  JSONOut.attributeObject(FieldName, [&] {
    JSONOut.attribute("line", Position.line());
    JSONOut.attribute("character", Position.column());
  });
}

void writeMetaTuple(LSIFMeta &Meta, const clang::clangd::IndexFileOut &O) {
  writeVertex(Meta, "metaData", [&](JOStream &JSONOut) {
    JSONOut.attribute("version", "0.4.3");
    JSONOut.attribute("projectRoot", Meta.ProjectRoot);
    JSONOut.attribute("positionEncoding", "utf-16");
    JSONOut.attributeObject("toolInfo", [&] {
      JSONOut.attribute("name", "lsif-clang");
      JSONOut.attribute("version", "0.1.0");
    });
  });
}

void writeProjectTuple(LSIFMeta &Meta, const clang::clangd::IndexFileOut &O) {
  writeVertex(Meta, "project",
              [&](JOStream &JSONOut) { JSONOut.attribute("kind", "cpp"); });
}

void writeDocumentTuple(LSIFMeta &Meta, const llvm::StringRef FileURI,
                        const std::string &Lang) {
  int DocumentID = writeVertex(Meta, "document", [&](JOStream &JSONOut) {
    JSONOut.attribute("uri", FileURI);
    JSONOut.attribute("languageId", Lang);
  });
  Meta.DocumentIDs[FileURI] = DocumentID;
}

int writeRange(LSIFMeta &Meta, const clang::clangd::SymbolLocation &Location,
               int ResultSetID, int ReferencesResultID, std::string Type) {
  if (!Meta.contains(Location.FileURI))
    return -1;
  int RangeID;
  if (Meta.RangeIDs.find(Location) == Meta.RangeIDs.end()) {
    RangeID = writeVertex(Meta, "range", [&](JOStream &JSONOut) {
      writePositionField(JSONOut, "start", Location.Start);
      writePositionField(JSONOut, "end", Location.End);
    });
    Meta.RangeIDs[Location] = RangeID;

    writeContainsEdge(Meta, Meta.DocumentIDs[Location.FileURI], {RangeID});
  } else {
    RangeID = Meta.RangeIDs[Location];
  }

  write1to1Edge(Meta, "next", RangeID, ResultSetID);

  writeItemEdge(Meta, ReferencesResultID, {RangeID},
                Meta.DocumentIDs[Location.FileURI], Type);

  return RangeID;
}

int writeRange(LSIFMeta &Meta, const clang::clangd::SymbolLocation &Location,
               const std::string &Lang, int ResultSetID, int ReferencesResultID,
               std::string Type) {
  if (!Meta.contains(Location.FileURI))
    return -1;
  if (Meta.DocumentIDs.find(Location.FileURI) == Meta.DocumentIDs.end()) {
    writeDocumentTuple(Meta, Location.FileURI, Lang);
  }
  return writeRange(Meta, Location, ResultSetID, ReferencesResultID, Type);
}

int writeResultSet(LSIFMeta &Meta) {
  return writeVertex(Meta, "resultSet", [&](JOStream &JSONOut) {});
}

std::string lang(const clang::clangd::Symbol &Sym) {
  switch (Sym.SymInfo.Lang) {
  case clang::index::SymbolLanguage::C:
    return "c";
  case clang::index::SymbolLanguage::CXX:
    return "cpp";
  case clang::index::SymbolLanguage::ObjC:
    return "objc";
  case clang::index::SymbolLanguage::Swift:
    return "swift";
  }
  return "cpp";
}

void writeHoverResult(LSIFMeta &Meta, const clang::clangd::Symbol &Sym,
                      int ResultSetID) {
  int HoverResultID = writeVertex(Meta, "hoverResult", [&](JOStream &JSONOut) {
    JSONOut.attributeObject("result", [&] {
      JSONOut.attributeArray("contents", [&] {
        JSONOut.object([&] {
          JSONOut.attribute("language", lang(Sym));

          std::ostringstream OHover;
          if (Sym.ReturnType.data() && !Sym.ReturnType.empty())
            OHover << Sym.ReturnType.data() << " ";
          if (Sym.Scope.data() && !Sym.Scope.empty())
            OHover << Sym.Scope.data();
          OHover << Sym.Name.data();
          if (Sym.Signature.data() && !Sym.Signature.empty())
            OHover << Sym.Signature.data();

          JSONOut.attribute("value", OHover.str());
        });
        if (Sym.Documentation.data() && !Sym.Documentation.empty()) {
          auto docString = Sym.Documentation;
          if(!llvm::json::isUTF8(docString)) {
            docString = llvm::StringRef(llvm::json::fixUTF8(Sym.Documentation));
          }
          JSONOut.value(docString);
        }
      });
    });
  });

  write1to1Edge(Meta, "textDocument/hover", ResultSetID, HoverResultID);
}

void writeDefinitionResult(LSIFMeta &Meta, const clang::clangd::Symbol &Sym,
                           int ResultSetID, int ReferencesResultID) {
  int DefinitionResultID =
      writeVertex(Meta, "definitionResult", [&](JOStream &JSONOut) {});
  write1to1Edge(Meta, "textDocument/definition", ResultSetID,
                DefinitionResultID);

  int DefinitionRangeID =
      writeRange(Meta, Sym.Definition, lang(Sym), ResultSetID,
                 ReferencesResultID, "definitions");
  writeItemEdge(Meta, DefinitionResultID, {DefinitionRangeID},
                Meta.DocumentIDs[Sym.Definition.FileURI]);
}

void writeSymbol(LSIFMeta &Meta, const clang::clangd::Symbol &Sym,
                 const clang::ArrayRef<clang::clangd::Ref> &Refs) {
  bool ContainsDef = Sym.Definition && Meta.contains(Sym.Definition.FileURI);
  // bool ContainsDecl = Sym.CanonicalDeclaration &&
  //                     Meta.contains(Sym.CanonicalDeclaration.FileURI);
  bool ContainsRef = false;
  for (const auto &Ref : Refs)
    if (Meta.contains(Ref.Location.FileURI))
      ContainsRef = true;

  if (!ContainsDef && !ContainsRef)
    return;

  int ResultSetID;
  int ReferenceResultID;
  if (Meta.ResultSetIDs.find(Sym.ID) == Meta.ResultSetIDs.end()) {
    ResultSetID = writeResultSet(Meta);
    Meta.ResultSetIDs[Sym.ID] = ResultSetID;
    ReferenceResultID =
        writeVertex(Meta, "referenceResult", [&](JOStream &JSONOut) {});
    Meta.ReferenceResultIDs[Sym.ID] = ReferenceResultID;

    writeHoverResult(Meta, Sym, ResultSetID);
    write1to1Edge(Meta, "textDocument/references", ResultSetID,
                  ReferenceResultID);

    if (ContainsDef) {
      writeDefinitionResult(Meta, Sym, ResultSetID, ReferenceResultID);
    }
  } else {
    ResultSetID = Meta.ResultSetIDs[Sym.ID];
    ReferenceResultID = Meta.ReferenceResultIDs[Sym.ID];
  }

  for (const auto &Ref : Refs) {
    if (!Meta.contains(Ref.Location.FileURI))
      continue;
    writeRange(Meta, Ref.Location, lang(Sym), ResultSetID, ReferenceResultID,
               "references");
  }
}

} // namespace

namespace clang {
namespace clangd {
void writeLSIF(const IndexFileOut &O, llvm::raw_ostream &OS) {
  LSIFMeta Meta(OS, O);

  writeMetaTuple(Meta, O);
  writeProjectTuple(Meta, O);

  // if (O.Sources) {
  //   for (const auto &Entry : *O.Sources) {
  //     writeDocumentTuple(Meta, Entry.second.URI);
  //   }
  // }

  for (const auto &Sym : *O.Symbols) {
    writeSymbol(Meta, Sym, {});
  }

  if (O.Refs) {
    for (const std::pair<SymbolID, ArrayRef<Ref>> &Ref : *O.Refs) {
      auto Sym = O.Symbols->find(Ref.first);
      if (Sym != O.Symbols->end()) {
        writeSymbol(Meta, *Sym, Ref.second);
      }
    }
  }

  Meta.OS.flush();
}
} // namespace clangd
} // namespace clang
