//===-- YAMLSerialization.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A YAML index file is a sequence of tagged entries.
// Each entry either encodes a Symbol or the list of references to a symbol
// (a "ref bundle").
//
//===----------------------------------------------------------------------===//

#include "Headers.h"
#include "index/Ref.h"
#include "index/Relation.h"
#include "index/Serialization.h"
#include "index/Symbol.h"
#include "index/SymbolLocation.h"
#include "index/SymbolOrigin.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <optional>

namespace {
struct YIncludeHeaderWithReferences;
}

LLVM_YAML_IS_SEQUENCE_VECTOR(clang::clangd::Symbol::IncludeHeaderWithReferences)
LLVM_YAML_IS_SEQUENCE_VECTOR(clang::clangd::Ref)
LLVM_YAML_IS_SEQUENCE_VECTOR(YIncludeHeaderWithReferences)

namespace {
using RefBundle =
    std::pair<clang::clangd::SymbolID, std::vector<clang::clangd::Ref>>;
// This is a pale imitation of std::variant<Symbol, RefBundle, Relation>
struct VariantEntry {
  std::optional<clang::clangd::Symbol> Symbol;
  std::optional<RefBundle> Refs;
  std::optional<clang::clangd::Relation> Relation;
  std::optional<clang::clangd::IncludeGraphNode> Source;
  std::optional<clang::tooling::CompileCommand> Cmd;
};
// A class helps YAML to serialize the 32-bit encoded position (Line&Column),
// as YAMLIO can't directly map bitfields.
struct YPosition {
  uint32_t Line;
  uint32_t Column;
};
// A class helps YAML to serialize the IncludeHeaderWithReferences as YAMLIO
// can't directly map bitfields.
struct YIncludeHeaderWithReferences {
  llvm::StringRef IncludeHeader;
  uint32_t References;
  clang::clangd::Symbol::IncludeDirective SupportedDirectives;

  YIncludeHeaderWithReferences() = default;

  YIncludeHeaderWithReferences(
      llvm::StringRef IncludeHeader, uint32_t References,
      clang::clangd::Symbol::IncludeDirective SupportedDirectives)
      : IncludeHeader(IncludeHeader), References(References),
        SupportedDirectives(SupportedDirectives) {}
};

// avoid ODR violation of specialization for non-owned CompileCommand
struct CompileCommandYAML : clang::tooling::CompileCommand {};

} // namespace
namespace llvm {
namespace yaml {

using clang::clangd::FileDigest;
using clang::clangd::IncludeGraph;
using clang::clangd::IncludeGraphNode;
using clang::clangd::Ref;
using clang::clangd::RefKind;
using clang::clangd::Relation;
using clang::clangd::RelationKind;
using clang::clangd::Symbol;
using clang::clangd::SymbolID;
using clang::clangd::SymbolLocation;
using clang::index::SymbolInfo;
using clang::index::SymbolKind;
using clang::index::SymbolLanguage;
using clang::tooling::CompileCommand;

// Helper to (de)serialize the SymbolID. We serialize it as a hex string.
struct NormalizedSymbolID {
  NormalizedSymbolID(IO &) {}
  NormalizedSymbolID(IO &, const SymbolID &ID) {
    llvm::raw_string_ostream OS(HexString);
    OS << ID;
  }

  SymbolID denormalize(IO &I) {
    auto ID = SymbolID::fromStr(HexString);
    if (!ID) {
      I.setError(llvm::toString(ID.takeError()));
      return SymbolID();
    }
    return *ID;
  }

  std::string HexString;
};

struct NormalizedSymbolFlag {
  NormalizedSymbolFlag(IO &) {}
  NormalizedSymbolFlag(IO &, Symbol::SymbolFlag F) {
    Flag = static_cast<uint8_t>(F);
  }

  Symbol::SymbolFlag denormalize(IO &) {
    return static_cast<Symbol::SymbolFlag>(Flag);
  }

  uint8_t Flag = 0;
};

template <> struct MappingTraits<YPosition> {
  static void mapping(IO &IO, YPosition &Value) {
    IO.mapRequired("Line", Value.Line);
    IO.mapRequired("Column", Value.Column);
  }
};

struct NormalizedPosition {
  using Position = clang::clangd::SymbolLocation::Position;
  NormalizedPosition(IO &) {}
  NormalizedPosition(IO &, const Position &Pos) {
    P.Line = Pos.line();
    P.Column = Pos.column();
  }

  Position denormalize(IO &) {
    Position Pos;
    Pos.setLine(P.Line);
    Pos.setColumn(P.Column);
    return Pos;
  }
  YPosition P;
};

struct NormalizedFileURI {
  NormalizedFileURI(IO &) {}
  NormalizedFileURI(IO &, const char *FileURI) { URI = FileURI; }

  const char *denormalize(IO &IO) {
    assert(IO.getContext() &&
           "Expecting an UniqueStringSaver to allocate data");
    return static_cast<llvm::UniqueStringSaver *>(IO.getContext())
        ->save(URI)
        .data();
  }

  std::string URI;
};

template <> struct MappingTraits<SymbolLocation> {
  static void mapping(IO &IO, SymbolLocation &Value) {
    MappingNormalization<NormalizedFileURI, const char *> NFile(IO,
                                                                Value.FileURI);
    IO.mapRequired("FileURI", NFile->URI);
    MappingNormalization<NormalizedPosition, SymbolLocation::Position> NStart(
        IO, Value.Start);
    IO.mapRequired("Start", NStart->P);
    MappingNormalization<NormalizedPosition, SymbolLocation::Position> NEnd(
        IO, Value.End);
    IO.mapRequired("End", NEnd->P);
  }
};

template <> struct MappingTraits<SymbolInfo> {
  static void mapping(IO &IO, SymbolInfo &SymInfo) {
    // FIXME: expose other fields?
    IO.mapRequired("Kind", SymInfo.Kind);
    IO.mapRequired("Lang", SymInfo.Lang);
  }
};

template <> struct ScalarBitSetTraits<clang::clangd::Symbol::IncludeDirective> {
  static void bitset(IO &IO, clang::clangd::Symbol::IncludeDirective &Value) {
    IO.bitSetCase(Value, "Include", clang::clangd::Symbol::Include);
    IO.bitSetCase(Value, "Import", clang::clangd::Symbol::Import);
  }
};

template <> struct MappingTraits<YIncludeHeaderWithReferences> {
  static void mapping(IO &IO, YIncludeHeaderWithReferences &Inc) {
    IO.mapRequired("Header", Inc.IncludeHeader);
    IO.mapRequired("References", Inc.References);
    IO.mapOptional("Directives", Inc.SupportedDirectives,
                   clang::clangd::Symbol::Include);
  }
};

struct NormalizedIncludeHeaders {
  using IncludeHeader = clang::clangd::Symbol::IncludeHeaderWithReferences;
  NormalizedIncludeHeaders(IO &) {}
  NormalizedIncludeHeaders(
      IO &, const llvm::SmallVector<IncludeHeader, 1> &IncludeHeaders) {
    for (auto &I : IncludeHeaders) {
      Headers.emplace_back(I.IncludeHeader, I.References,
                           I.supportedDirectives());
    }
  }

  llvm::SmallVector<IncludeHeader, 1> denormalize(IO &) {
    llvm::SmallVector<IncludeHeader, 1> Result;
    for (auto &H : Headers)
      Result.emplace_back(H.IncludeHeader, H.References, H.SupportedDirectives);
    return Result;
  }
  llvm::SmallVector<YIncludeHeaderWithReferences, 1> Headers;
};

template <> struct MappingTraits<Symbol> {
  static void mapping(IO &IO, Symbol &Sym) {
    MappingNormalization<NormalizedSymbolID, SymbolID> NSymbolID(IO, Sym.ID);
    MappingNormalization<NormalizedSymbolFlag, Symbol::SymbolFlag> NSymbolFlag(
        IO, Sym.Flags);
    MappingNormalization<
        NormalizedIncludeHeaders,
        llvm::SmallVector<Symbol::IncludeHeaderWithReferences, 1>>
        NIncludeHeaders(IO, Sym.IncludeHeaders);
    IO.mapRequired("ID", NSymbolID->HexString);
    IO.mapRequired("Name", Sym.Name);
    IO.mapRequired("Scope", Sym.Scope);
    IO.mapRequired("SymInfo", Sym.SymInfo);
    IO.mapOptional("CanonicalDeclaration", Sym.CanonicalDeclaration,
                   SymbolLocation());
    IO.mapOptional("Definition", Sym.Definition, SymbolLocation());
    IO.mapOptional("References", Sym.References, 0u);
    IO.mapOptional("Flags", NSymbolFlag->Flag);
    IO.mapOptional("Signature", Sym.Signature);
    IO.mapOptional("TemplateSpecializationArgs",
                   Sym.TemplateSpecializationArgs);
    IO.mapOptional("CompletionSnippetSuffix", Sym.CompletionSnippetSuffix);
    IO.mapOptional("Documentation", Sym.Documentation);
    IO.mapOptional("ReturnType", Sym.ReturnType);
    IO.mapOptional("Type", Sym.Type);
    IO.mapOptional("IncludeHeaders", NIncludeHeaders->Headers);
  }
};

template <> struct ScalarEnumerationTraits<SymbolLanguage> {
  static void enumeration(IO &IO, SymbolLanguage &Value) {
    IO.enumCase(Value, "C", SymbolLanguage::C);
    IO.enumCase(Value, "Cpp", SymbolLanguage::CXX);
    IO.enumCase(Value, "ObjC", SymbolLanguage::ObjC);
    IO.enumCase(Value, "Swift", SymbolLanguage::Swift);
  }
};

template <> struct ScalarEnumerationTraits<SymbolKind> {
  static void enumeration(IO &IO, SymbolKind &Value) {
#define DEFINE_ENUM(name) IO.enumCase(Value, #name, SymbolKind::name)

    DEFINE_ENUM(Unknown);
    DEFINE_ENUM(Function);
    DEFINE_ENUM(Module);
    DEFINE_ENUM(Namespace);
    DEFINE_ENUM(NamespaceAlias);
    DEFINE_ENUM(Macro);
    DEFINE_ENUM(Enum);
    DEFINE_ENUM(Struct);
    DEFINE_ENUM(Class);
    DEFINE_ENUM(Protocol);
    DEFINE_ENUM(Extension);
    DEFINE_ENUM(Union);
    DEFINE_ENUM(TypeAlias);
    DEFINE_ENUM(Function);
    DEFINE_ENUM(Variable);
    DEFINE_ENUM(Field);
    DEFINE_ENUM(EnumConstant);
    DEFINE_ENUM(InstanceMethod);
    DEFINE_ENUM(ClassMethod);
    DEFINE_ENUM(StaticMethod);
    DEFINE_ENUM(InstanceProperty);
    DEFINE_ENUM(ClassProperty);
    DEFINE_ENUM(StaticProperty);
    DEFINE_ENUM(Constructor);
    DEFINE_ENUM(Destructor);
    DEFINE_ENUM(ConversionFunction);
    DEFINE_ENUM(Parameter);
    DEFINE_ENUM(Using);

#undef DEFINE_ENUM
  }
};

template <> struct MappingTraits<RefBundle> {
  static void mapping(IO &IO, RefBundle &Refs) {
    MappingNormalization<NormalizedSymbolID, SymbolID> NSymbolID(IO,
                                                                 Refs.first);
    IO.mapRequired("ID", NSymbolID->HexString);
    IO.mapRequired("References", Refs.second);
  }
};

struct NormalizedRefKind {
  NormalizedRefKind(IO &) {}
  NormalizedRefKind(IO &, RefKind O) { Kind = static_cast<uint8_t>(O); }

  RefKind denormalize(IO &) { return static_cast<RefKind>(Kind); }

  uint8_t Kind = 0;
};

template <> struct MappingTraits<Ref> {
  static void mapping(IO &IO, Ref &R) {
    MappingNormalization<NormalizedRefKind, RefKind> NKind(IO, R.Kind);
    IO.mapRequired("Kind", NKind->Kind);
    IO.mapRequired("Location", R.Location);
  }
};

struct NormalizedSymbolRole {
  NormalizedSymbolRole(IO &) {}
  NormalizedSymbolRole(IO &IO, RelationKind R) {
    Kind = static_cast<uint8_t>(R);
  }

  RelationKind denormalize(IO &IO) { return static_cast<RelationKind>(Kind); }

  uint8_t Kind = 0;
};

template <> struct MappingTraits<SymbolID> {
  static void mapping(IO &IO, SymbolID &ID) {
    MappingNormalization<NormalizedSymbolID, SymbolID> NSymbolID(IO, ID);
    IO.mapRequired("ID", NSymbolID->HexString);
  }
};

template <> struct MappingTraits<Relation> {
  static void mapping(IO &IO, Relation &Relation) {
    MappingNormalization<NormalizedSymbolRole, RelationKind> NRole(
        IO, Relation.Predicate);
    IO.mapRequired("Subject", Relation.Subject);
    IO.mapRequired("Predicate", NRole->Kind);
    IO.mapRequired("Object", Relation.Object);
  }
};

struct NormalizedSourceFlag {
  NormalizedSourceFlag(IO &) {}
  NormalizedSourceFlag(IO &, IncludeGraphNode::SourceFlag O) {
    Flag = static_cast<uint8_t>(O);
  }

  IncludeGraphNode::SourceFlag denormalize(IO &) {
    return static_cast<IncludeGraphNode::SourceFlag>(Flag);
  }

  uint8_t Flag = 0;
};

struct NormalizedFileDigest {
  NormalizedFileDigest(IO &) {}
  NormalizedFileDigest(IO &, const FileDigest &Digest) {
    HexString = llvm::toHex(Digest);
  }

  FileDigest denormalize(IO &I) {
    FileDigest Digest;
    if (HexString.size() == Digest.size() * 2 &&
        llvm::all_of(HexString, llvm::isHexDigit)) {
      memcpy(Digest.data(), llvm::fromHex(HexString).data(), Digest.size());
    } else {
      I.setError(std::string("Bad hex file digest: ") + HexString);
    }
    return Digest;
  }

  std::string HexString;
};

template <> struct MappingTraits<IncludeGraphNode> {
  static void mapping(IO &IO, IncludeGraphNode &Node) {
    IO.mapRequired("URI", Node.URI);
    MappingNormalization<NormalizedSourceFlag, IncludeGraphNode::SourceFlag>
        NSourceFlag(IO, Node.Flags);
    IO.mapRequired("Flags", NSourceFlag->Flag);
    MappingNormalization<NormalizedFileDigest, FileDigest> NDigest(IO,
                                                                   Node.Digest);
    IO.mapRequired("Digest", NDigest->HexString);
    IO.mapRequired("DirectIncludes", Node.DirectIncludes);
  }
};

template <> struct MappingTraits<CompileCommandYAML> {
  static void mapping(IO &IO, CompileCommandYAML &Cmd) {
    IO.mapRequired("Directory", Cmd.Directory);
    IO.mapRequired("CommandLine", Cmd.CommandLine);
  }
};

template <> struct MappingTraits<VariantEntry> {
  static void mapping(IO &IO, VariantEntry &Variant) {
    if (IO.mapTag("!Symbol", Variant.Symbol.has_value())) {
      if (!IO.outputting())
        Variant.Symbol.emplace();
      MappingTraits<Symbol>::mapping(IO, *Variant.Symbol);
    } else if (IO.mapTag("!Refs", Variant.Refs.has_value())) {
      if (!IO.outputting())
        Variant.Refs.emplace();
      MappingTraits<RefBundle>::mapping(IO, *Variant.Refs);
    } else if (IO.mapTag("!Relations", Variant.Relation.has_value())) {
      if (!IO.outputting())
        Variant.Relation.emplace();
      MappingTraits<Relation>::mapping(IO, *Variant.Relation);
    } else if (IO.mapTag("!Source", Variant.Source.has_value())) {
      if (!IO.outputting())
        Variant.Source.emplace();
      MappingTraits<IncludeGraphNode>::mapping(IO, *Variant.Source);
    } else if (IO.mapTag("!Cmd", Variant.Cmd.has_value())) {
      if (!IO.outputting())
        Variant.Cmd.emplace();
      MappingTraits<CompileCommandYAML>::mapping(
          IO, static_cast<CompileCommandYAML &>(*Variant.Cmd));
    }
  }
};

} // namespace yaml
} // namespace llvm

namespace clang {
namespace clangd {

void writeYAML(const IndexFileOut &O, llvm::raw_ostream &OS) {
  llvm::yaml::Output Yout(OS);
  for (const auto &Sym : *O.Symbols) {
    VariantEntry Entry;
    Entry.Symbol = Sym;
    Yout << Entry;
  }
  if (O.Refs)
    for (auto &Sym : *O.Refs) {
      VariantEntry Entry;
      Entry.Refs = Sym;
      Yout << Entry;
    }
  if (O.Relations)
    for (auto &R : *O.Relations) {
      VariantEntry Entry;
      Entry.Relation = R;
      Yout << Entry;
    }
  if (O.Sources) {
    for (const auto &Source : *O.Sources) {
      VariantEntry Entry;
      Entry.Source = Source.getValue();
      Yout << Entry;
    }
  }
  if (O.Cmd) {
    VariantEntry Entry;
    Entry.Cmd = *O.Cmd;
    Yout << Entry;
  }
}

llvm::Expected<IndexFileIn> readYAML(llvm::StringRef Data,
                                     SymbolOrigin Origin) {
  SymbolSlab::Builder Symbols;
  RefSlab::Builder Refs;
  RelationSlab::Builder Relations;
  llvm::BumpPtrAllocator
      Arena; // store the underlying data of Position::FileURI.
  llvm::UniqueStringSaver Strings(Arena);
  llvm::yaml::Input Yin(Data, &Strings);
  IncludeGraph Sources;
  std::optional<tooling::CompileCommand> Cmd;
  while (Yin.setCurrentDocument()) {
    llvm::yaml::EmptyContext Ctx;
    VariantEntry Variant;
    yamlize(Yin, Variant, true, Ctx);
    if (Yin.error())
      return llvm::errorCodeToError(Yin.error());

    if (Variant.Symbol) {
      Variant.Symbol->Origin = Origin;
      Symbols.insert(*Variant.Symbol);
    }
    if (Variant.Refs)
      for (const auto &Ref : Variant.Refs->second)
        Refs.insert(Variant.Refs->first, Ref);
    if (Variant.Relation)
      Relations.insert(*Variant.Relation);
    if (Variant.Source) {
      auto &IGN = *Variant.Source;
      auto Entry = Sources.try_emplace(IGN.URI).first;
      Entry->getValue() = std::move(IGN);
      // Fixup refs to refer to map keys which will live on
      Entry->getValue().URI = Entry->getKey();
      for (auto &Include : Entry->getValue().DirectIncludes)
        Include = Sources.try_emplace(Include).first->getKey();
    }
    if (Variant.Cmd)
      Cmd = *Variant.Cmd;
    Yin.nextDocument();
  }

  IndexFileIn Result;
  Result.Symbols.emplace(std::move(Symbols).build());
  Result.Refs.emplace(std::move(Refs).build());
  Result.Relations.emplace(std::move(Relations).build());
  if (Sources.size())
    Result.Sources = std::move(Sources);
  Result.Cmd = std::move(Cmd);
  return std::move(Result);
}

std::string toYAML(const Symbol &S) {
  std::string Buf;
  {
    llvm::raw_string_ostream OS(Buf);
    llvm::yaml::Output Yout(OS);
    Symbol Sym = S; // copy: Yout<< requires mutability.
    Yout << Sym;
  }
  return Buf;
}

std::string toYAML(const std::pair<SymbolID, llvm::ArrayRef<Ref>> &Data) {
  RefBundle Refs = {Data.first, Data.second};
  std::string Buf;
  {
    llvm::raw_string_ostream OS(Buf);
    llvm::yaml::Output Yout(OS);
    Yout << Refs;
  }
  return Buf;
}

std::string toYAML(const Relation &R) {
  std::string Buf;
  {
    llvm::raw_string_ostream OS(Buf);
    llvm::yaml::Output Yout(OS);
    Relation Rel = R; // copy: Yout<< requires mutability.
    Yout << Rel;
  }
  return Buf;
}

std::string toYAML(const Ref &R) {
  std::string Buf;
  {
    llvm::raw_string_ostream OS(Buf);
    llvm::yaml::Output Yout(OS);
    Ref Reference = R; // copy: Yout<< requires mutability.
    Yout << Reference;
  }
  return Buf;
}

} // namespace clangd
} // namespace clang
