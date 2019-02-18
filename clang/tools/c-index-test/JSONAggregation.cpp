//===--- JSONAggregation.cpp - Index data aggregation in JSON format ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "JSONAggregation.h"
#include "indexstore/IndexStoreCXX.h"
#include "clang/Frontend/Utils.h"
#include "clang/Index/IndexDataStoreSymbolUtils.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/BuryPointer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::index;
using namespace indexstore;
using namespace llvm;

namespace {

typedef size_t FilePathIndex;
typedef size_t RecordIndex;
typedef size_t SymbolIndex;

struct UnitSourceInfo {
  FilePathIndex FilePath;
  SmallVector<RecordIndex, 2> AssociatedRecords;
};

struct UnitInfo {
  std::string Name;
  SmallVector<UnitSourceInfo, 8> Sources;
  SmallVector<std::string, 3> UnitDepends;
  FilePathIndex OutFile;
  StringRef Triple;
};

struct SymbolInfo {
  SymbolKind Kind;
  SymbolLanguage Lang;
  StringRef USR;
  StringRef Name;
  StringRef CodegenName;
  SymbolRoleSet Roles = 0;
  SymbolRoleSet RelatedRoles = 0;
};

struct SymbolRelationInfo {
  SymbolIndex RelatedSymbol;
  SymbolRoleSet Roles;
  SymbolRelationInfo(SymbolIndex relSymbol, SymbolRoleSet roles)
    : RelatedSymbol(relSymbol), Roles(roles) {}
};

struct SymbolOccurrenceInfo {
  SymbolIndex Symbol;
  SymbolRoleSet Roles = 0;
  std::vector<SymbolRelationInfo> Relations;
  unsigned Line;
  unsigned Column;
};

struct RecordInfo {
  SmallVector<SymbolOccurrenceInfo, 8> Occurrences;
};

class Aggregator {
  IndexStore Store;

  BumpPtrAllocator Allocator;

  StringMap<FilePathIndex, BumpPtrAllocator &> FilePathIndices;
  std::vector<StringRef> FilePaths;
  StringMap<char, BumpPtrAllocator &> Triples;

  std::vector<std::unique_ptr<UnitInfo>> Units;

  StringMap<RecordIndex, BumpPtrAllocator &> RecordIndices;
  std::vector<std::unique_ptr<RecordInfo>> Records;

  StringMap<SymbolIndex, BumpPtrAllocator &> SymbolIndices;
  std::vector<SymbolInfo> Symbols;

public:
  explicit Aggregator(IndexStore store)
  : Store(std::move(store)),
    FilePathIndices(Allocator),
    Triples(Allocator),
    RecordIndices(Allocator),
    SymbolIndices(Allocator) {}

  bool process();
  void processUnit(StringRef name, IndexUnitReader &UnitReader);
  void dumpJSON(raw_ostream &OS);

private:
  StringRef copyStr(StringRef str) {
    if (str.empty())
      return StringRef();
    char *buf = Allocator.Allocate<char>(str.size());
    std::copy(str.begin(), str.end(), buf);
    return StringRef(buf, str.size());
  }

  StringRef getTripleString(StringRef inputTriple) {
    return Triples.insert(std::make_pair(inputTriple, 0)).first->first();
  }

  FilePathIndex getFilePathIndex(StringRef path, StringRef workingDir);
  RecordIndex getRecordIndex(StringRef recordFile);
  SymbolIndex getSymbolIndex(IndexRecordSymbol sym);
  std::unique_ptr<RecordInfo> processRecord(StringRef recordFile);
};

} // anonymous namespace

bool Aggregator::process() {
  bool succ = Store.foreachUnit(/*sorted=*/true, [&](StringRef unitName) -> bool {
    std::string error;
    auto unitReader = IndexUnitReader(Store, unitName, error);
    if (!unitReader) {
      errs() << "error opening unit file '" << unitName << "': " << error << '\n';
      return false;
    }

    processUnit(unitName, unitReader);
    return true;
  });

  return !succ;
}

void Aggregator::processUnit(StringRef name, IndexUnitReader &UnitReader) {
  auto workDir = UnitReader.getWorkingDirectory();
  auto unit = llvm::make_unique<UnitInfo>();
  unit->Name = name;
  unit->Triple = getTripleString(UnitReader.getTarget());
  unit->OutFile = getFilePathIndex(UnitReader.getOutputFile(), workDir);

  struct DepInfo {
    UnitSourceInfo source;
    std::string unitName;
  };
  SmallVector<DepInfo, 32> Deps;
  UnitReader.foreachDependency([&](IndexUnitDependency dep) -> bool {
    Deps.resize(Deps.size()+1);
    auto &depInfo = Deps.back();
    switch (dep.getKind()) {
      case IndexUnitDependency::DependencyKind::Unit: {
        depInfo.unitName = dep.getName();
        StringRef filePath = dep.getFilePath();
        if (!filePath.empty())
          depInfo.source.FilePath = getFilePathIndex(filePath, workDir);
        break;
      }
      case IndexUnitDependency::DependencyKind::Record: {
        depInfo.source.FilePath = getFilePathIndex(dep.getFilePath(), workDir);
        RecordIndex recIndex = getRecordIndex(dep.getName());
        depInfo.source.AssociatedRecords.push_back(recIndex);
        break;
      }
      case IndexUnitDependency::DependencyKind::File:
        depInfo.source.FilePath = getFilePathIndex(dep.getFilePath(), workDir);
    }
    return true;
  });

  unit->Sources.reserve(Deps.size());
  for (auto &dep : Deps) {
    if (!dep.unitName.empty()) {
      unit->UnitDepends.emplace_back(std::move(dep.unitName));
    } else {
      unit->Sources.push_back(std::move(dep.source));
    }
  }

  Units.push_back(std::move(unit));
}

FilePathIndex Aggregator::getFilePathIndex(StringRef path, StringRef workingDir) {
  StringRef absPath;
  SmallString<128> absPathBuf;
  if (sys::path::is_absolute(path) || workingDir.empty()) {
    absPath = path;
  } else {
    absPathBuf = workingDir;
    sys::path::append(absPathBuf, path);
    absPath = absPathBuf.str();
  }

  auto pair = FilePathIndices.insert(std::make_pair(absPath, FilePaths.size()));
  bool wasInserted = pair.second;
  if (wasInserted) {
    FilePaths.push_back(pair.first->first());
  }
  return pair.first->second;
}

RecordIndex Aggregator::getRecordIndex(StringRef recordFile) {
  auto pair = RecordIndices.insert(std::make_pair(recordFile, Records.size()));
  bool wasInserted = pair.second;
  if (wasInserted) {
    Records.push_back(processRecord(recordFile));
  }
  return pair.first->second;
}

std::unique_ptr<RecordInfo> Aggregator::processRecord(StringRef recordFile) {
  std::string error;
  auto recordReader = IndexRecordReader(Store, recordFile, error);
  if (!recordReader) {
    errs() << "failed reading record file: " << recordFile << '\n';
    ::exit(1);
  }
  auto record = llvm::make_unique<RecordInfo>();
  recordReader.foreachOccurrence([&](IndexRecordOccurrence idxOccur) -> bool {
    SymbolIndex symIdx = getSymbolIndex(idxOccur.getSymbol());
    SymbolInfo &symInfo = Symbols[symIdx];
    symInfo.Roles |= getSymbolRoles(idxOccur.getRoles());
    SymbolOccurrenceInfo occurInfo;
    occurInfo.Symbol = symIdx;
    idxOccur.foreachRelation([&](IndexSymbolRelation rel) -> bool {
      SymbolIndex relsymIdx = getSymbolIndex(rel.getSymbol());
      SymbolInfo &relsymInfo = Symbols[relsymIdx];
      relsymInfo.RelatedRoles |= getSymbolRoles(rel.getRoles());
      occurInfo.Relations.emplace_back(relsymIdx, getSymbolRoles(rel.getRoles()));
      return true;
    });
    occurInfo.Roles = getSymbolRoles(idxOccur.getRoles());
    std::tie(occurInfo.Line, occurInfo.Column) = idxOccur.getLineCol();
    record->Occurrences.push_back(std::move(occurInfo));
    return true;
  });
  return record;
}

SymbolIndex Aggregator::getSymbolIndex(IndexRecordSymbol sym) {
  auto pair = SymbolIndices.insert(std::make_pair(sym.getUSR(), Symbols.size()));
  bool wasInserted = pair.second;
  if (wasInserted) {
    SymbolInfo symInfo;
    symInfo.Kind = getSymbolKind(sym.getKind());
    symInfo.Lang = getSymbolLanguage(sym.getLanguage());
    symInfo.USR = pair.first->first();
    symInfo.Name = copyStr(sym.getName());
    symInfo.CodegenName = copyStr(sym.getCodegenName());
    Symbols.push_back(std::move(symInfo));
  }
  return pair.first->second;
}


void Aggregator::dumpJSON(raw_ostream &OS) {
  OS << "{\n";
  OS.indent(2) << "\"files\": [\n";
  for (unsigned i = 0, e = FilePaths.size(); i != e; ++i) {
    OS.indent(4) << '\"' << FilePaths[i] << '\"';
    if (i < e-1) OS << ',';
    OS << '\n';
  }
  OS.indent(2) << "],\n";

  OS.indent(2) << "\"symbols\": [\n";
  for (unsigned i = 0, e = Symbols.size(); i != e; ++i) {
    OS.indent(4) << "{\n";
    SymbolInfo &symInfo = Symbols[i];
    OS.indent(6) << "\"kind\": \"" << getSymbolKindString(symInfo.Kind) << "\",\n";
    OS.indent(6) << "\"lang\": \"" << getSymbolLanguageString(symInfo.Lang) << "\",\n";
    OS.indent(6) << "\"usr\": \"" << symInfo.USR << "\",\n";
    OS.indent(6) << "\"name\": \"" << symInfo.Name << "\",\n";
    if (!symInfo.CodegenName.empty())
      OS.indent(6) << "\"codegen\": \"" << symInfo.CodegenName << "\",\n";
    OS.indent(6) << "\"roles\": \"";
    printSymbolRoles(symInfo.Roles, OS);
    OS << '\"';
    if (symInfo.RelatedRoles != 0) {
      OS << ",\n";
      OS.indent(6) << "\"rel-roles\": \"";
      printSymbolRoles(symInfo.RelatedRoles, OS);
      OS << '\"';
    }
    OS << '\n';
    OS.indent(4) << "}";
    if (i < e-1) OS << ',';
    OS << '\n';
  }
  OS.indent(2) << "],\n";

  OS.indent(2) << "\"records\": [\n";
  for (unsigned i = 0, e = Records.size(); i != e; ++i) {
    OS.indent(4) << "{\n";
    RecordInfo &recInfo = *Records[i];
    OS.indent(6) << "\"occurrences\": [\n";
    for (unsigned oi = 0, oe = recInfo.Occurrences.size(); oi != oe; ++oi) {
      OS.indent(8) << "{\n";
      SymbolOccurrenceInfo &occurInfo = recInfo.Occurrences[oi];
      OS.indent(10) << "\"symbol\": " << occurInfo.Symbol << ",\n";
      OS.indent(10) << "\"line\": " << occurInfo.Line << ",\n";
      OS.indent(10) << "\"col\": " << occurInfo.Column << ",\n";
      OS.indent(10) << "\"roles\": \"";
      printSymbolRoles(occurInfo.Roles, OS);
      OS << '\"';
      if (!occurInfo.Relations.empty()) {
        OS << ",\n";
        OS.indent(10) << "\"relations\": [\n";
        for (unsigned ri = 0, re = occurInfo.Relations.size(); ri != re; ++ri) {
          OS.indent(12) << "{\n";
          SymbolRelationInfo &relInfo = occurInfo.Relations[ri];
          OS.indent(14) << "\"symbol\": " << relInfo.RelatedSymbol << ",\n";
          OS.indent(14) << "\"rel-roles\": \"";
          printSymbolRoles(relInfo.Roles, OS);
          OS << "\"\n";
          OS.indent(12) << "}";
          if (ri < re-1) OS << ',';
          OS << '\n';
        }
        OS.indent(10) << "]\n";
      }
      OS << '\n';
      OS.indent(8) << "}";
      if (oi < oe-1) OS << ',';
      OS << '\n';
    }
    OS.indent(6) << "]\n";
    OS.indent(4) << "}";
    if (i < e-1) OS << ',';
    OS << '\n';
  }
  OS.indent(2) << "],\n";

  StringMap<size_t> UnitIndicesByName;
  for (unsigned i = 0, e = Units.size(); i != e; ++i) {
    UnitInfo &unit = *Units[i];
    UnitIndicesByName[unit.Name] = i;
  }

  OS.indent(2) << "\"units\": [\n";
  for (unsigned i = 0, e = Units.size(); i != e; ++i) {
    OS.indent(4) << "{\n";
    UnitInfo &unit = *Units[i];
    OS.indent(6) << "\"triple\": \"" << unit.Triple << "\",\n";
    OS.indent(6) << "\"out-file\": " << unit.OutFile << ",\n";
    if (!unit.UnitDepends.empty()) {
      OS.indent(6) << "\"unit-dependencies\": [";
      for (unsigned ui = 0, ue = unit.UnitDepends.size(); ui != ue; ++ui) {
        OS << UnitIndicesByName[unit.UnitDepends[ui]];
        if (ui < ue-1) OS << ", ";
      }
      OS << "],\n";
    }
    OS.indent(6) << "\"sources\": [\n";
    for (unsigned si = 0, se = unit.Sources.size(); si != se; ++si) {
      OS.indent(8) << "{\n";
      UnitSourceInfo &source = unit.Sources[si];
      OS.indent(10) << "\"file\": " << source.FilePath;
      if (!source.AssociatedRecords.empty()) {
        OS << ",\n";
        OS.indent(10) << "\"records\": [";
        for (unsigned ri = 0, re = source.AssociatedRecords.size(); ri != re; ++ri) {
          OS << source.AssociatedRecords[ri];
          if (ri < re-1) OS << ", ";
        }
        OS << ']';
      }
      OS << '\n';
      OS.indent(8) << "}";
      if (si < se-1) OS << ',';
      OS << '\n';
    }
    OS.indent(6) << "]\n";
    OS.indent(4) << "}";
    if (i < e-1) OS << ',';
    OS << '\n';
  }
  OS.indent(2) << "]\n";
  OS << "}\n";
}


bool index::aggregateDataAsJSON(StringRef StorePath, raw_ostream &OS) {
  std::string error;
  auto dataStore = IndexStore(StorePath, error);
  if (!dataStore) {
    errs() << "error opening store path '" << StorePath << "': " << error << '\n';
    return true;
  }

  // Explicitely avoid doing any memory cleanup for aggregator since the process
  // is going to exit when we are done.
  Aggregator *aggregator = new Aggregator(std::move(dataStore));
  bool err = aggregator->process();
  if (err)
    return true;
  aggregator->dumpJSON(OS);
  llvm::BuryPointer(aggregator);
  return false;
}
