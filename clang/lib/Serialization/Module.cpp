//===--- Module.cpp - Module description ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the Module class, which describes a module that has
//  been loaded from an AST file.
//
//===----------------------------------------------------------------------===//
#include "clang/Serialization/Module.h"
#include "ASTReaderInternals.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace serialization;
using namespace reader;

ModuleFile::ModuleFile(ModuleKind Kind, unsigned Generation)
  : Kind(Kind), File(nullptr), Signature(0), DirectlyImported(false),
    Generation(Generation), SizeInBits(0),
    LocalNumSLocEntries(0), SLocEntryBaseID(0),
    SLocEntryBaseOffset(0), SLocEntryOffsets(nullptr),
    LocalNumIdentifiers(0),
    IdentifierOffsets(nullptr), BaseIdentifierID(0),
    IdentifierTableData(nullptr), IdentifierLookupTable(nullptr),
    LocalNumMacros(0), MacroOffsets(nullptr),
    BasePreprocessedEntityID(0),
    PreprocessedEntityOffsets(nullptr), NumPreprocessedEntities(0),
    LocalNumHeaderFileInfos(0), 
    HeaderFileInfoTableData(nullptr), HeaderFileInfoTable(nullptr),
    LocalNumSubmodules(0), BaseSubmoduleID(0),
    LocalNumSelectors(0), SelectorOffsets(nullptr), BaseSelectorID(0),
    SelectorLookupTableData(nullptr), SelectorLookupTable(nullptr),
    LocalNumDecls(0), DeclOffsets(nullptr), BaseDeclID(0),
    FileSortedDecls(nullptr), NumFileSortedDecls(0),
    ObjCCategoriesMap(nullptr), LocalNumObjCCategoriesInMap(0),
    LocalNumTypes(0), TypeOffsets(nullptr), BaseTypeIndex(0)
{}

ModuleFile::~ModuleFile() {
  delete static_cast<ASTIdentifierLookupTable *>(IdentifierLookupTable);
  delete static_cast<HeaderFileInfoLookupTable *>(HeaderFileInfoTable);
  delete static_cast<ASTSelectorLookupTable *>(SelectorLookupTable);
}

template<typename Key, typename Offset, unsigned InitialCapacity>
static void 
dumpLocalRemap(StringRef Name,
               const ContinuousRangeMap<Key, Offset, InitialCapacity> &Map) {
  if (Map.begin() == Map.end())
    return;
  
  typedef ContinuousRangeMap<Key, Offset, InitialCapacity> MapType;
  llvm::errs() << "  " << Name << ":\n";
  for (typename MapType::const_iterator I = Map.begin(), IEnd = Map.end(); 
       I != IEnd; ++I) {
    llvm::errs() << "    " << I->first << " -> " << I->second << "\n";
  }
}

LLVM_DUMP_METHOD void ModuleFile::dump() {
  llvm::errs() << "\nModule: " << FileName << "\n";
  if (!Imports.empty()) {
    llvm::errs() << "  Imports: ";
    for (unsigned I = 0, N = Imports.size(); I != N; ++I) {
      if (I)
        llvm::errs() << ", ";
      llvm::errs() << Imports[I]->FileName;
    }
    llvm::errs() << "\n";
  }
  
  // Remapping tables.
  llvm::errs() << "  Base source location offset: " << SLocEntryBaseOffset 
               << '\n';
  dumpLocalRemap("Source location offset local -> global map", SLocRemap);
  
  llvm::errs() << "  Base identifier ID: " << BaseIdentifierID << '\n'
               << "  Number of identifiers: " << LocalNumIdentifiers << '\n';
  dumpLocalRemap("Identifier ID local -> global map", IdentifierRemap);

  llvm::errs() << "  Base macro ID: " << BaseMacroID << '\n'
               << "  Number of macros: " << LocalNumMacros << '\n';
  dumpLocalRemap("Macro ID local -> global map", MacroRemap);

  llvm::errs() << "  Base submodule ID: " << BaseSubmoduleID << '\n'
               << "  Number of submodules: " << LocalNumSubmodules << '\n';
  dumpLocalRemap("Submodule ID local -> global map", SubmoduleRemap);

  llvm::errs() << "  Base selector ID: " << BaseSelectorID << '\n'
               << "  Number of selectors: " << LocalNumSelectors << '\n';
  dumpLocalRemap("Selector ID local -> global map", SelectorRemap);
  
  llvm::errs() << "  Base preprocessed entity ID: " << BasePreprocessedEntityID
               << '\n'  
               << "  Number of preprocessed entities: " 
               << NumPreprocessedEntities << '\n';
  dumpLocalRemap("Preprocessed entity ID local -> global map", 
                 PreprocessedEntityRemap);
  
  llvm::errs() << "  Base type index: " << BaseTypeIndex << '\n'
               << "  Number of types: " << LocalNumTypes << '\n';
  dumpLocalRemap("Type index local -> global map", TypeRemap);
  
  llvm::errs() << "  Base decl ID: " << BaseDeclID << '\n'
               << "  Number of decls: " << LocalNumDecls << '\n';
  dumpLocalRemap("Decl ID local -> global map", DeclRemap);
}
