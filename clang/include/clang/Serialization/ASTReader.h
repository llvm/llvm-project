//===--- ASTReader.h - AST File Reader --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ASTReader class, which reads AST files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_AST_READER_H
#define LLVM_CLANG_FRONTEND_AST_READER_H

#include "clang/Serialization/ASTBitCodes.h"
#include "clang/Serialization/ContinuousRangeMap.h"
#include "clang/Serialization/Module.h"
#include "clang/Serialization/ModuleManager.h"
#include "clang/Sema/ExternalSemaSource.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/TemplateBase.h"
#include "clang/Lex/ExternalPreprocessorSource.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/PPMutationListener.h"
#include "clang/Lex/PreprocessingRecord.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Bitcode/BitstreamReader.h"
#include "llvm/Support/DataTypes.h"
#include <deque>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace llvm {
  class MemoryBuffer;
}

namespace clang {

class AddrLabelExpr;
class ASTConsumer;
class ASTContext;
class ASTIdentifierIterator;
class ASTUnit; // FIXME: Layering violation and egregious hack.
class Attr;
class Decl;
class DeclContext;
class DiagnosticOptions;
class NestedNameSpecifier;
class CXXBaseSpecifier;
class CXXConstructorDecl;
class CXXCtorInitializer;
class GotoStmt;
class MacroDefinition;
class NamedDecl;
class OpaqueValueExpr;
class Preprocessor;
class PreprocessorOptions;
class Sema;
class SwitchCase;
class ASTDeserializationListener;
class ASTWriter;
class ASTReader;
class ASTDeclReader;
class ASTStmtReader;
class TypeLocReader;
struct HeaderFileInfo;
class VersionTuple;
class TargetOptions;

/// \brief Abstract interface for callback invocations by the ASTReader.
///
/// While reading an AST file, the ASTReader will call the methods of the
/// listener to pass on specific information. Some of the listener methods can
/// return true to indicate to the ASTReader that the information (and
/// consequently the AST file) is invalid.
class ASTReaderListener {
public:
  virtual ~ASTReaderListener();

  /// \brief Receives the language options.
  ///
  /// \returns true to indicate the options are invalid or false otherwise.
  virtual bool ReadLanguageOptions(const LangOptions &LangOpts,
                                   bool Complain) {
    return false;
  }

  /// \brief Receives the target options.
  ///
  /// \returns true to indicate the target options are invalid, or false
  /// otherwise.
  virtual bool ReadTargetOptions(const TargetOptions &TargetOpts,
                                 bool Complain) {
    return false;
  }

  /// \brief Receives the diagnostic options.
  ///
  /// \returns true to indicate the diagnostic options are invalid, or false
  /// otherwise.
  virtual bool ReadDiagnosticOptions(const DiagnosticOptions &DiagOpts,
                                     bool Complain) {
    return false;
  }

  /// \brief Receives the file system options.
  ///
  /// \returns true to indicate the file system options are invalid, or false
  /// otherwise.
  virtual bool ReadFileSystemOptions(const FileSystemOptions &FSOpts,
                                     bool Complain) {
    return false;
  }

  /// \brief Receives the header search options.
  ///
  /// \returns true to indicate the header search options are invalid, or false
  /// otherwise.
  virtual bool ReadHeaderSearchOptions(const HeaderSearchOptions &HSOpts,
                                       bool Complain) {
    return false;
  }

  /// \brief Receives the preprocessor options.
  ///
  /// \param SuggestedPredefines Can be filled in with the set of predefines
  /// that are suggested by the preprocessor options. Typically only used when
  /// loading a precompiled header.
  ///
  /// \returns true to indicate the preprocessor options are invalid, or false
  /// otherwise.
  virtual bool ReadPreprocessorOptions(const PreprocessorOptions &PPOpts,
                                       bool Complain,
                                       std::string &SuggestedPredefines) {
    return false;
  }

  /// \brief Receives a HeaderFileInfo entry.
  virtual void ReadHeaderFileInfo(const HeaderFileInfo &HFI, unsigned ID) {}

  /// \brief Receives __COUNTER__ value.
  virtual void ReadCounter(const serialization::ModuleFile &M,
                           unsigned Value) {}
};

/// \brief ASTReaderListener implementation to validate the information of
/// the PCH file against an initialized Preprocessor.
class PCHValidator : public ASTReaderListener {
  Preprocessor &PP;
  ASTReader &Reader;

  unsigned NumHeaderInfos;

public:
  PCHValidator(Preprocessor &PP, ASTReader &Reader)
    : PP(PP), Reader(Reader), NumHeaderInfos(0) {}

  virtual bool ReadLanguageOptions(const LangOptions &LangOpts,
                                   bool Complain);
  virtual bool ReadTargetOptions(const TargetOptions &TargetOpts,
                                 bool Complain);
  virtual bool ReadPreprocessorOptions(const PreprocessorOptions &PPOpts,
                                       bool Complain,
                                       std::string &SuggestedPredefines);
  virtual void ReadHeaderFileInfo(const HeaderFileInfo &HFI, unsigned ID);
  virtual void ReadCounter(const serialization::ModuleFile &M, unsigned Value);

private:
  void Error(const char *Msg);
};

namespace serialization {

class ReadMethodPoolVisitor;

namespace reader {
  class ASTIdentifierLookupTrait;
  /// \brief The on-disk hash table used for the DeclContext's Name lookup table.
  typedef OnDiskChainedHashTable<ASTDeclContextNameLookupTrait>
    ASTDeclContextNameLookupTable;
}

} // end namespace serialization

/// \brief Reads an AST files chain containing the contents of a translation
/// unit.
///
/// The ASTReader class reads bitstreams (produced by the ASTWriter
/// class) containing the serialized representation of a given
/// abstract syntax tree and its supporting data structures. An
/// instance of the ASTReader can be attached to an ASTContext object,
/// which will provide access to the contents of the AST files.
///
/// The AST reader provides lazy de-serialization of declarations, as
/// required when traversing the AST. Only those AST nodes that are
/// actually required will be de-serialized.
class ASTReader
  : public ExternalPreprocessorSource,
    public ExternalPreprocessingRecordSource,
    public ExternalHeaderFileInfoSource,
    public ExternalSemaSource,
    public IdentifierInfoLookup,
    public ExternalIdentifierLookup,
    public ExternalSLocEntrySource
{
public:
  typedef SmallVector<uint64_t, 64> RecordData;

  /// \brief The result of reading the control block of an AST file, which
  /// can fail for various reasons.
  enum ASTReadResult {
    /// \brief The control block was read successfully. Aside from failures,
    /// the AST file is safe to read into the current context.
    Success,
    /// \brief The AST file itself appears corrupted.
    Failure,
    /// \brief The AST file is out-of-date relative to its input files,
    /// and needs to be regenerated.
    OutOfDate,
    /// \brief The AST file was written by a different version of Clang.
    VersionMismatch,
    /// \brief The AST file was writtten with a different language/target
    /// configuration.
    ConfigurationMismatch,
    /// \brief The AST file has errors.
    HadErrors
  };
  
  /// \brief Types of AST files.
  friend class PCHValidator;
  friend class ASTDeclReader;
  friend class ASTStmtReader;
  friend class ASTIdentifierIterator;
  friend class serialization::reader::ASTIdentifierLookupTrait;
  friend class TypeLocReader;
  friend class ASTWriter;
  friend class ASTUnit; // ASTUnit needs to remap source locations.
  friend class serialization::ReadMethodPoolVisitor;

  typedef serialization::ModuleFile ModuleFile;
  typedef serialization::ModuleKind ModuleKind;
  typedef serialization::ModuleManager ModuleManager;

  typedef ModuleManager::ModuleIterator ModuleIterator;
  typedef ModuleManager::ModuleConstIterator ModuleConstIterator;
  typedef ModuleManager::ModuleReverseIterator ModuleReverseIterator;

private:
  /// \brief The receiver of some callbacks invoked by ASTReader.
  OwningPtr<ASTReaderListener> Listener;

  /// \brief The receiver of deserialization events.
  ASTDeserializationListener *DeserializationListener;

  SourceManager &SourceMgr;
  FileManager &FileMgr;
  DiagnosticsEngine &Diags;

  /// \brief The semantic analysis object that will be processing the
  /// AST files and the translation unit that uses it.
  Sema *SemaObj;

  /// \brief The preprocessor that will be loading the source file.
  Preprocessor &PP;

  /// \brief The AST context into which we'll read the AST files.
  ASTContext &Context;

  /// \brief The AST consumer.
  ASTConsumer *Consumer;

  /// \brief The module manager which manages modules and their dependencies
  ModuleManager ModuleMgr;

  /// \brief A map of global bit offsets to the module that stores entities
  /// at those bit offsets.
  ContinuousRangeMap<uint64_t, ModuleFile*, 4> GlobalBitOffsetsMap;

  /// \brief A map of negated SLocEntryIDs to the modules containing them.
  ContinuousRangeMap<unsigned, ModuleFile*, 64> GlobalSLocEntryMap;

  typedef ContinuousRangeMap<unsigned, ModuleFile*, 64> GlobalSLocOffsetMapType;

  /// \brief A map of reversed (SourceManager::MaxLoadedOffset - SLocOffset)
  /// SourceLocation offsets to the modules containing them.
  GlobalSLocOffsetMapType GlobalSLocOffsetMap;

  /// \brief Types that have already been loaded from the chain.
  ///
  /// When the pointer at index I is non-NULL, the type with
  /// ID = (I + 1) << FastQual::Width has already been loaded
  std::vector<QualType> TypesLoaded;

  typedef ContinuousRangeMap<serialization::TypeID, ModuleFile *, 4>
    GlobalTypeMapType;

  /// \brief Mapping from global type IDs to the module in which the
  /// type resides along with the offset that should be added to the
  /// global type ID to produce a local ID.
  GlobalTypeMapType GlobalTypeMap;

  /// \brief Declarations that have already been loaded from the chain.
  ///
  /// When the pointer at index I is non-NULL, the declaration with ID
  /// = I + 1 has already been loaded.
  std::vector<Decl *> DeclsLoaded;

  typedef ContinuousRangeMap<serialization::DeclID, ModuleFile *, 4>
    GlobalDeclMapType;

  /// \brief Mapping from global declaration IDs to the module in which the
  /// declaration resides.
  GlobalDeclMapType GlobalDeclMap;

  typedef std::pair<ModuleFile *, uint64_t> FileOffset;
  typedef SmallVector<FileOffset, 2> FileOffsetsTy;
  typedef llvm::DenseMap<serialization::DeclID, FileOffsetsTy>
      DeclUpdateOffsetsMap;

  /// \brief Declarations that have modifications residing in a later file
  /// in the chain.
  DeclUpdateOffsetsMap DeclUpdateOffsets;

  struct ReplacedDeclInfo {
    ModuleFile *Mod;
    uint64_t Offset;
    unsigned RawLoc;

    ReplacedDeclInfo() : Mod(0), Offset(0), RawLoc(0) {}
    ReplacedDeclInfo(ModuleFile *Mod, uint64_t Offset, unsigned RawLoc)
      : Mod(Mod), Offset(Offset), RawLoc(RawLoc) {}
  };

  typedef llvm::DenseMap<serialization::DeclID, ReplacedDeclInfo>
      DeclReplacementMap;
  /// \brief Declarations that have been replaced in a later file in the chain.
  DeclReplacementMap ReplacedDecls;

  struct FileDeclsInfo {
    ModuleFile *Mod;
    ArrayRef<serialization::LocalDeclID> Decls;

    FileDeclsInfo() : Mod(0) {}
    FileDeclsInfo(ModuleFile *Mod, ArrayRef<serialization::LocalDeclID> Decls)
      : Mod(Mod), Decls(Decls) {}
  };

  /// \brief Map from a FileID to the file-level declarations that it contains.
  llvm::DenseMap<FileID, FileDeclsInfo> FileDeclIDs;

  // Updates for visible decls can occur for other contexts than just the
  // TU, and when we read those update records, the actual context will not
  // be available yet (unless it's the TU), so have this pending map using the
  // ID as a key. It will be realized when the context is actually loaded.
  typedef
    SmallVector<std::pair<serialization::reader::ASTDeclContextNameLookupTable *,
                          ModuleFile*>, 1> DeclContextVisibleUpdates;
  typedef llvm::DenseMap<serialization::DeclID, DeclContextVisibleUpdates>
      DeclContextVisibleUpdatesPending;

  /// \brief Updates to the visible declarations of declaration contexts that
  /// haven't been loaded yet.
  DeclContextVisibleUpdatesPending PendingVisibleUpdates;
  
  /// \brief The set of C++ or Objective-C classes that have forward 
  /// declarations that have not yet been linked to their definitions.
  llvm::SmallPtrSet<Decl *, 4> PendingDefinitions;

  typedef llvm::MapVector<Decl *, uint64_t,
                          llvm::SmallDenseMap<Decl *, unsigned, 4>,
                          llvm::SmallVector<std::pair<Decl *, uint64_t>, 4> >
    PendingBodiesMap;

  /// \brief Functions or methods that have bodies that will be attached.
  PendingBodiesMap PendingBodies;

  /// \brief Read the records that describe the contents of declcontexts.
  bool ReadDeclContextStorage(ModuleFile &M,
                              llvm::BitstreamCursor &Cursor,
                              const std::pair<uint64_t, uint64_t> &Offsets,
                              serialization::DeclContextInfo &Info);

  /// \brief A vector containing identifiers that have already been
  /// loaded.
  ///
  /// If the pointer at index I is non-NULL, then it refers to the
  /// IdentifierInfo for the identifier with ID=I+1 that has already
  /// been loaded.
  std::vector<IdentifierInfo *> IdentifiersLoaded;

  typedef ContinuousRangeMap<serialization::IdentID, ModuleFile *, 4>
    GlobalIdentifierMapType;

  /// \brief Mapping from global identifier IDs to the module in which the
  /// identifier resides along with the offset that should be added to the
  /// global identifier ID to produce a local ID.
  GlobalIdentifierMapType GlobalIdentifierMap;

  /// \brief A vector containing macros that have already been
  /// loaded.
  ///
  /// If the pointer at index I is non-NULL, then it refers to the
  /// MacroInfo for the identifier with ID=I+1 that has already
  /// been loaded.
  std::vector<MacroInfo *> MacrosLoaded;

  typedef ContinuousRangeMap<serialization::MacroID, ModuleFile *, 4>
    GlobalMacroMapType;

  /// \brief Mapping from global macro IDs to the module in which the
  /// macro resides along with the offset that should be added to the
  /// global macro ID to produce a local ID.
  GlobalMacroMapType GlobalMacroMap;

  typedef llvm::DenseMap<serialization::MacroID,
            llvm::SmallVector<std::pair<serialization::SubmoduleID,
                                        MacroUpdate>, 1> >
    MacroUpdatesMap;

  /// \brief Mapping from (global) macro IDs to the set of updates to be
  /// performed to the corresponding macro.
  MacroUpdatesMap MacroUpdates;

  /// \brief A vector containing submodules that have already been loaded.
  ///
  /// This vector is indexed by the Submodule ID (-1). NULL submodule entries
  /// indicate that the particular submodule ID has not yet been loaded.
  SmallVector<Module *, 2> SubmodulesLoaded;
  
  typedef ContinuousRangeMap<serialization::SubmoduleID, ModuleFile *, 4>
    GlobalSubmoduleMapType;
  
  /// \brief Mapping from global submodule IDs to the module file in which the
  /// submodule resides along with the offset that should be added to the
  /// global submodule ID to produce a local ID.
  GlobalSubmoduleMapType GlobalSubmoduleMap;

  /// \brief An entity that has been hidden.
  class HiddenName {
  public:
    enum NameKind {
      Declaration,
      MacroVisibility,
      MacroUndef
    } Kind;

  private:
    unsigned Loc;

    union {
      Decl *D;
      MacroInfo *MI;
    };

    IdentifierInfo *Id;

  public:
    HiddenName(Decl *D) : Kind(Declaration), Loc(), D(D), Id() { }

    HiddenName(IdentifierInfo *II, MacroInfo *MI)
      : Kind(MacroVisibility), Loc(), MI(MI), Id(II) { }

    HiddenName(IdentifierInfo *II, MacroInfo *MI, SourceLocation Loc)
      : Kind(MacroUndef), Loc(Loc.getRawEncoding()), MI(MI), Id(II) { }

    NameKind getKind() const { return Kind; }

    Decl *getDecl() const {
      assert(getKind() == Declaration && "Hidden name is not a declaration");
      return D;
    }

    std::pair<IdentifierInfo *, MacroInfo *> getMacro() const {
      assert((getKind() == MacroUndef || getKind() == MacroVisibility)
             && "Hidden name is not a macro!");
      return std::make_pair(Id, MI);
    }

    SourceLocation getMacroUndefLoc() const {
      assert(getKind() == MacroUndef && "Hidden name is not an undef!");
      return SourceLocation::getFromRawEncoding(Loc);
    }
};

  /// \brief A set of hidden declarations.
  typedef llvm::SmallVector<HiddenName, 2>
    HiddenNames;
  
  typedef llvm::DenseMap<Module *, HiddenNames> HiddenNamesMapType;

  /// \brief A mapping from each of the hidden submodules to the deserialized
  /// declarations in that submodule that could be made visible.
  HiddenNamesMapType HiddenNamesMap;
  
  
  /// \brief A module import or export that hasn't yet been resolved.
  struct UnresolvedModuleImportExport {
    /// \brief The file in which this module resides.
    ModuleFile *File;
    
    /// \brief The module that is importing or exporting.
    Module *Mod;
    
    /// \brief The local ID of the module that is being exported.
    unsigned ID;
    
    /// \brief Whether this is an import (vs. an export).
    unsigned IsImport : 1;
    
    /// \brief Whether this is a wildcard export.
    unsigned IsWildcard : 1;
  };
  
  /// \brief The set of module imports and exports that still need to be 
  /// resolved.
  llvm::SmallVector<UnresolvedModuleImportExport, 2> 
    UnresolvedModuleImportExports;
  
  /// \brief A vector containing selectors that have already been loaded.
  ///
  /// This vector is indexed by the Selector ID (-1). NULL selector
  /// entries indicate that the particular selector ID has not yet
  /// been loaded.
  SmallVector<Selector, 16> SelectorsLoaded;

  typedef ContinuousRangeMap<serialization::SelectorID, ModuleFile *, 4>
    GlobalSelectorMapType;

  /// \brief Mapping from global selector IDs to the module in which the
  /// selector resides along with the offset that should be added to the
  /// global selector ID to produce a local ID.
  GlobalSelectorMapType GlobalSelectorMap;

  /// \brief The generation number of the last time we loaded data from the
  /// global method pool for this selector.
  llvm::DenseMap<Selector, unsigned> SelectorGeneration;

  typedef llvm::MapVector<IdentifierInfo *,
                          llvm::SmallVector<serialization::MacroID, 2> >
    PendingMacroIDsMap;

  /// \brief Mapping from identifiers that have a macro history to the global
  /// IDs have not yet been deserialized to the global IDs of those macros.
  PendingMacroIDsMap PendingMacroIDs;

  typedef ContinuousRangeMap<unsigned, ModuleFile *, 4>
    GlobalPreprocessedEntityMapType;

  /// \brief Mapping from global preprocessing entity IDs to the module in
  /// which the preprocessed entity resides along with the offset that should be
  /// added to the global preprocessing entitiy ID to produce a local ID.
  GlobalPreprocessedEntityMapType GlobalPreprocessedEntityMap;

  /// \name CodeGen-relevant special data
  /// \brief Fields containing data that is relevant to CodeGen.
  //@{

  /// \brief The IDs of all declarations that fulfill the criteria of
  /// "interesting" decls.
  ///
  /// This contains the data loaded from all EXTERNAL_DEFINITIONS blocks in the
  /// chain. The referenced declarations are deserialized and passed to the
  /// consumer eagerly.
  SmallVector<uint64_t, 16> ExternalDefinitions;

  /// \brief The IDs of all tentative definitions stored in the chain.
  ///
  /// Sema keeps track of all tentative definitions in a TU because it has to
  /// complete them and pass them on to CodeGen. Thus, tentative definitions in
  /// the PCH chain must be eagerly deserialized.
  SmallVector<uint64_t, 16> TentativeDefinitions;

  /// \brief The IDs of all CXXRecordDecls stored in the chain whose VTables are
  /// used.
  ///
  /// CodeGen has to emit VTables for these records, so they have to be eagerly
  /// deserialized.
  SmallVector<uint64_t, 64> VTableUses;

  /// \brief A snapshot of the pending instantiations in the chain.
  ///
  /// This record tracks the instantiations that Sema has to perform at the
  /// end of the TU. It consists of a pair of values for every pending
  /// instantiation where the first value is the ID of the decl and the second
  /// is the instantiation location.
  SmallVector<uint64_t, 64> PendingInstantiations;

  //@}

  /// \name DiagnosticsEngine-relevant special data
  /// \brief Fields containing data that is used for generating diagnostics
  //@{

  /// \brief A snapshot of Sema's unused file-scoped variable tracking, for
  /// generating warnings.
  SmallVector<uint64_t, 16> UnusedFileScopedDecls;

  /// \brief A list of all the delegating constructors we've seen, to diagnose
  /// cycles.
  SmallVector<uint64_t, 4> DelegatingCtorDecls;

  /// \brief Method selectors used in a @selector expression. Used for
  /// implementation of -Wselector.
  SmallVector<uint64_t, 64> ReferencedSelectorsData;

  /// \brief A snapshot of Sema's weak undeclared identifier tracking, for
  /// generating warnings.
  SmallVector<uint64_t, 64> WeakUndeclaredIdentifiers;

  /// \brief The IDs of type aliases for ext_vectors that exist in the chain.
  ///
  /// Used by Sema for finding sugared names for ext_vectors in diagnostics.
  SmallVector<uint64_t, 4> ExtVectorDecls;

  //@}

  /// \name Sema-relevant special data
  /// \brief Fields containing data that is used for semantic analysis
  //@{

  /// \brief The IDs of all locally scoped external decls in the chain.
  ///
  /// Sema tracks these to validate that the types are consistent across all
  /// local external declarations.
  SmallVector<uint64_t, 16> LocallyScopedExternalDecls;

  /// \brief The IDs of all dynamic class declarations in the chain.
  ///
  /// Sema tracks these because it checks for the key functions being defined
  /// at the end of the TU, in which case it directs CodeGen to emit the VTable.
  SmallVector<uint64_t, 16> DynamicClasses;

  /// \brief The IDs of the declarations Sema stores directly.
  ///
  /// Sema tracks a few important decls, such as namespace std, directly.
  SmallVector<uint64_t, 4> SemaDeclRefs;

  /// \brief The IDs of the types ASTContext stores directly.
  ///
  /// The AST context tracks a few important types, such as va_list, directly.
  SmallVector<uint64_t, 16> SpecialTypes;

  /// \brief The IDs of CUDA-specific declarations ASTContext stores directly.
  ///
  /// The AST context tracks a few important decls, currently cudaConfigureCall,
  /// directly.
  SmallVector<uint64_t, 2> CUDASpecialDeclRefs;

  /// \brief The floating point pragma option settings.
  SmallVector<uint64_t, 1> FPPragmaOptions;

  /// \brief The OpenCL extension settings.
  SmallVector<uint64_t, 1> OpenCLExtensions;

  /// \brief A list of the namespaces we've seen.
  SmallVector<uint64_t, 4> KnownNamespaces;

  /// \brief A list of modules that were imported by precompiled headers or
  /// any other non-module AST file.
  SmallVector<serialization::SubmoduleID, 2> ImportedModules;
  //@}

  /// \brief The directory that the PCH we are reading is stored in.
  std::string CurrentDir;

  /// \brief The system include root to be used when loading the
  /// precompiled header.
  std::string isysroot;

  /// \brief Whether to disable the normal validation performed on precompiled
  /// headers when they are loaded.
  bool DisableValidation;

  /// \brief Whether to accept an AST file with compiler errors.
  bool AllowASTWithCompilerErrors;

  /// \brief The current "generation" of the module file import stack, which 
  /// indicates how many separate module file load operations have occurred.
  unsigned CurrentGeneration;

  typedef llvm::DenseMap<unsigned, SwitchCase *> SwitchCaseMapTy;
  /// \brief Mapping from switch-case IDs in the chain to switch-case statements
  ///
  /// Statements usually don't have IDs, but switch cases need them, so that the
  /// switch statement can refer to them.
  SwitchCaseMapTy SwitchCaseStmts;

  SwitchCaseMapTy *CurrSwitchCaseStmts;

  /// \brief The number of source location entries de-serialized from
  /// the PCH file.
  unsigned NumSLocEntriesRead;

  /// \brief The number of source location entries in the chain.
  unsigned TotalNumSLocEntries;

  /// \brief The number of statements (and expressions) de-serialized
  /// from the chain.
  unsigned NumStatementsRead;

  /// \brief The total number of statements (and expressions) stored
  /// in the chain.
  unsigned TotalNumStatements;

  /// \brief The number of macros de-serialized from the chain.
  unsigned NumMacrosRead;

  /// \brief The total number of macros stored in the chain.
  unsigned TotalNumMacros;

  /// \brief The number of selectors that have been read.
  unsigned NumSelectorsRead;

  /// \brief The number of method pool entries that have been read.
  unsigned NumMethodPoolEntriesRead;

  /// \brief The number of times we have looked up a selector in the method
  /// pool and not found anything interesting.
  unsigned NumMethodPoolMisses;

  /// \brief The total number of method pool entries in the selector table.
  unsigned TotalNumMethodPoolEntries;

  /// Number of lexical decl contexts read/total.
  unsigned NumLexicalDeclContextsRead, TotalLexicalDeclContexts;

  /// Number of visible decl contexts read/total.
  unsigned NumVisibleDeclContextsRead, TotalVisibleDeclContexts;

  /// Total size of modules, in bits, currently loaded
  uint64_t TotalModulesSizeInBits;

  /// \brief Number of Decl/types that are currently deserializing.
  unsigned NumCurrentElementsDeserializing;

  /// \brief Set true while we are in the process of passing deserialized
  /// "interesting" decls to consumer inside FinishedDeserializing().
  /// This is used as a guard to avoid recursively repeating the process of
  /// passing decls to consumer.
  bool PassingDeclsToConsumer;

  /// Number of CXX base specifiers currently loaded
  unsigned NumCXXBaseSpecifiersLoaded;

  /// \brief An IdentifierInfo that has been loaded but whose top-level
  /// declarations of the same name have not (yet) been loaded.
  struct PendingIdentifierInfo {
    IdentifierInfo *II;
    SmallVector<uint32_t, 4> DeclIDs;
  };

  /// \brief The set of identifiers that were read while the AST reader was
  /// (recursively) loading declarations.
  ///
  /// The declarations on the identifier chain for these identifiers will be
  /// loaded once the recursive loading has completed.
  std::deque<PendingIdentifierInfo> PendingIdentifierInfos;

  /// \brief The generation number of each identifier, which keeps track of
  /// the last time we loaded information about this identifier.
  llvm::DenseMap<IdentifierInfo *, unsigned> IdentifierGeneration;
  
  /// \brief Contains declarations and definitions that will be
  /// "interesting" to the ASTConsumer, when we get that AST consumer.
  ///
  /// "Interesting" declarations are those that have data that may
  /// need to be emitted, such as inline function definitions or
  /// Objective-C protocols.
  std::deque<Decl *> InterestingDecls;

  /// \brief The set of redeclarable declarations that have been deserialized
  /// since the last time the declaration chains were linked.
  llvm::SmallPtrSet<Decl *, 16> RedeclsDeserialized;
  
  /// \brief The list of redeclaration chains that still need to be 
  /// reconstructed.
  ///
  /// Each element is the global declaration ID of the first declaration in
  /// the chain. Elements in this vector should be unique; use 
  /// PendingDeclChainsKnown to ensure uniqueness.
  llvm::SmallVector<serialization::DeclID, 16> PendingDeclChains;

  /// \brief Keeps track of the elements added to PendingDeclChains.
  llvm::SmallSet<serialization::DeclID, 16> PendingDeclChainsKnown;

  /// \brief The set of Objective-C categories that have been deserialized
  /// since the last time the declaration chains were linked.
  llvm::SmallPtrSet<ObjCCategoryDecl *, 16> CategoriesDeserialized;

  /// \brief The set of Objective-C class definitions that have already been
  /// loaded, for which we will need to check for categories whenever a new
  /// module is loaded.
  llvm::SmallVector<ObjCInterfaceDecl *, 16> ObjCClassesLoaded;
  
  typedef llvm::DenseMap<Decl *, llvm::SmallVector<serialization::DeclID, 2> >
    MergedDeclsMap;
    
  /// \brief A mapping from canonical declarations to the set of additional
  /// (global, previously-canonical) declaration IDs that have been merged with
  /// that canonical declaration.
  MergedDeclsMap MergedDecls;
  
  typedef llvm::DenseMap<serialization::GlobalDeclID, 
                         llvm::SmallVector<serialization::DeclID, 2> >
    StoredMergedDeclsMap;
  
  /// \brief A mapping from canonical declaration IDs to the set of additional
  /// declaration IDs that have been merged with that canonical declaration.
  ///
  /// This is the deserialized representation of the entries in MergedDecls.
  /// When we query entries in MergedDecls, they will be augmented with entries
  /// from StoredMergedDecls.
  StoredMergedDeclsMap StoredMergedDecls;
  
  /// \brief Combine the stored merged declarations for the given canonical
  /// declaration into the set of merged declarations.
  ///
  /// \returns An iterator into MergedDecls that corresponds to the position of
  /// the given canonical declaration.
  MergedDeclsMap::iterator
  combineStoredMergedDecls(Decl *Canon, serialization::GlobalDeclID CanonID);
  
  /// \brief Ready to load the previous declaration of the given Decl.
  void loadAndAttachPreviousDecl(Decl *D, serialization::DeclID ID);

  /// \brief When reading a Stmt tree, Stmt operands are placed in this stack.
  SmallVector<Stmt *, 16> StmtStack;

  /// \brief What kind of records we are reading.
  enum ReadingKind {
    Read_Decl, Read_Type, Read_Stmt
  };

  /// \brief What kind of records we are reading.
  ReadingKind ReadingKind;

  /// \brief RAII object to change the reading kind.
  class ReadingKindTracker {
    ASTReader &Reader;
    enum ReadingKind PrevKind;

    ReadingKindTracker(const ReadingKindTracker &) LLVM_DELETED_FUNCTION;
    void operator=(const ReadingKindTracker &) LLVM_DELETED_FUNCTION;

  public:
    ReadingKindTracker(enum ReadingKind newKind, ASTReader &reader)
      : Reader(reader), PrevKind(Reader.ReadingKind) {
      Reader.ReadingKind = newKind;
    }

    ~ReadingKindTracker() { Reader.ReadingKind = PrevKind; }
  };

  /// \brief Suggested contents of the predefines buffer, after this
  /// PCH file has been processed.
  ///
  /// In most cases, this string will be empty, because the predefines
  /// buffer computed to build the PCH file will be identical to the
  /// predefines buffer computed from the command line. However, when
  /// there are differences that the PCH reader can work around, this
  /// predefines buffer may contain additional definitions.
  std::string SuggestedPredefines;

  /// \brief Reads a statement from the specified cursor.
  Stmt *ReadStmtFromStream(ModuleFile &F);

  typedef llvm::PointerIntPair<const FileEntry *, 1, bool> InputFile;

  /// \brief Retrieve the file entry and 'overridden' bit for an input
  /// file in the given module file.
  InputFile getInputFile(ModuleFile &F, unsigned ID, bool Complain = true);

  /// \brief Get a FileEntry out of stored-in-PCH filename, making sure we take
  /// into account all the necessary relocations.
  const FileEntry *getFileEntry(StringRef filename);

  void MaybeAddSystemRootToFilename(ModuleFile &M, std::string &Filename);

  ASTReadResult ReadASTCore(StringRef FileName, ModuleKind Type,
                            ModuleFile *ImportedBy,
                            llvm::SmallVectorImpl<ModuleFile *> &Loaded,
                            unsigned ClientLoadCapabilities);
  ASTReadResult ReadControlBlock(ModuleFile &F,
                                 llvm::SmallVectorImpl<ModuleFile *> &Loaded,
                                 unsigned ClientLoadCapabilities);
  bool ReadASTBlock(ModuleFile &F);
  bool ParseLineTable(ModuleFile &F, SmallVectorImpl<uint64_t> &Record);
  bool ReadSourceManagerBlock(ModuleFile &F);
  llvm::BitstreamCursor &SLocCursorForID(int ID);
  SourceLocation getImportLocation(ModuleFile *F);
  bool ReadSubmoduleBlock(ModuleFile &F);
  static bool ParseLanguageOptions(const RecordData &Record, bool Complain,
                                   ASTReaderListener &Listener);
  static bool ParseTargetOptions(const RecordData &Record, bool Complain,
                                 ASTReaderListener &Listener);
  static bool ParseDiagnosticOptions(const RecordData &Record, bool Complain,
                                     ASTReaderListener &Listener);
  static bool ParseFileSystemOptions(const RecordData &Record, bool Complain,
                                     ASTReaderListener &Listener);
  static bool ParseHeaderSearchOptions(const RecordData &Record, bool Complain,
                                       ASTReaderListener &Listener);
  static bool ParsePreprocessorOptions(const RecordData &Record, bool Complain,
                                       ASTReaderListener &Listener,
                                       std::string &SuggestedPredefines);

  struct RecordLocation {
    RecordLocation(ModuleFile *M, uint64_t O)
      : F(M), Offset(O) {}
    ModuleFile *F;
    uint64_t Offset;
  };

  QualType readTypeRecord(unsigned Index);
  RecordLocation TypeCursorForIndex(unsigned Index);
  void LoadedDecl(unsigned Index, Decl *D);
  Decl *ReadDeclRecord(serialization::DeclID ID);
  RecordLocation DeclCursorForID(serialization::DeclID ID,
                                 unsigned &RawLocation);
  void loadDeclUpdateRecords(serialization::DeclID ID, Decl *D);
  void loadPendingDeclChain(serialization::GlobalDeclID ID);
  void loadObjCCategories(serialization::GlobalDeclID ID, ObjCInterfaceDecl *D,
                          unsigned PreviousGeneration = 0);

  RecordLocation getLocalBitOffset(uint64_t GlobalOffset);
  uint64_t getGlobalBitOffset(ModuleFile &M, uint32_t LocalOffset);

  /// \brief Returns the first preprocessed entity ID that ends after BLoc.
  serialization::PreprocessedEntityID
    findBeginPreprocessedEntity(SourceLocation BLoc) const;

  /// \brief Returns the first preprocessed entity ID that begins after ELoc.
  serialization::PreprocessedEntityID
    findEndPreprocessedEntity(SourceLocation ELoc) const;

  /// \brief Find the next module that contains entities and return the ID
  /// of the first entry.
  ///
  /// \param SLocMapI points at a chunk of a module that contains no
  /// preprocessed entities or the entities it contains are not the
  /// ones we are looking for.
  serialization::PreprocessedEntityID
    findNextPreprocessedEntity(
                        GlobalSLocOffsetMapType::const_iterator SLocMapI) const;

  /// \brief Returns (ModuleFile, Local index) pair for \p GlobalIndex of a
  /// preprocessed entity.
  std::pair<ModuleFile *, unsigned>
    getModulePreprocessedEntity(unsigned GlobalIndex);

  /// \brief Returns (begin, end) pair for the preprocessed entities of a
  /// particular module.
  std::pair<PreprocessingRecord::iterator, PreprocessingRecord::iterator>
    getModulePreprocessedEntities(ModuleFile &Mod) const;

  class ModuleDeclIterator {
    ASTReader *Reader;
    ModuleFile *Mod;
    const serialization::LocalDeclID *Pos;

  public:
    typedef const Decl *value_type;
    typedef value_type&         reference;
    typedef value_type*         pointer;

    ModuleDeclIterator() : Reader(0), Mod(0), Pos(0) { }

    ModuleDeclIterator(ASTReader *Reader, ModuleFile *Mod,
                       const serialization::LocalDeclID *Pos)
      : Reader(Reader), Mod(Mod), Pos(Pos) { }

    value_type operator*() const {
      return Reader->GetDecl(Reader->getGlobalDeclID(*Mod, *Pos));
    }

    ModuleDeclIterator &operator++() {
      ++Pos;
      return *this;
    }

    ModuleDeclIterator operator++(int) {
      ModuleDeclIterator Prev(*this);
      ++Pos;
      return Prev;
    }

    ModuleDeclIterator &operator--() {
      --Pos;
      return *this;
    }

    ModuleDeclIterator operator--(int) {
      ModuleDeclIterator Prev(*this);
      --Pos;
      return Prev;
    }

    friend bool operator==(const ModuleDeclIterator &LHS,
                           const ModuleDeclIterator &RHS) {
      assert(LHS.Reader == RHS.Reader && LHS.Mod == RHS.Mod);
      return LHS.Pos == RHS.Pos;
    }

    friend bool operator!=(const ModuleDeclIterator &LHS,
                           const ModuleDeclIterator &RHS) {
      assert(LHS.Reader == RHS.Reader && LHS.Mod == RHS.Mod);
      return LHS.Pos != RHS.Pos;
    }
  };

  std::pair<ModuleDeclIterator, ModuleDeclIterator>
    getModuleFileLevelDecls(ModuleFile &Mod);

  void PassInterestingDeclsToConsumer();
  void PassInterestingDeclToConsumer(Decl *D);

  void finishPendingActions();

  /// \brief Produce an error diagnostic and return true.
  ///
  /// This routine should only be used for fatal errors that have to
  /// do with non-routine failures (e.g., corrupted AST file).
  void Error(StringRef Msg);
  void Error(unsigned DiagID, StringRef Arg1 = StringRef(),
             StringRef Arg2 = StringRef());

  ASTReader(const ASTReader &) LLVM_DELETED_FUNCTION;
  void operator=(const ASTReader &) LLVM_DELETED_FUNCTION;
public:
  /// \brief Load the AST file and validate its contents against the given
  /// Preprocessor.
  ///
  /// \param PP the preprocessor associated with the context in which this
  /// precompiled header will be loaded.
  ///
  /// \param Context the AST context that this precompiled header will be
  /// loaded into.
  ///
  /// \param isysroot If non-NULL, the system include path specified by the
  /// user. This is only used with relocatable PCH files. If non-NULL,
  /// a relocatable PCH file will use the default path "/".
  ///
  /// \param DisableValidation If true, the AST reader will suppress most
  /// of its regular consistency checking, allowing the use of precompiled
  /// headers that cannot be determined to be compatible.
  ///
  /// \param AllowASTWithCompilerErrors If true, the AST reader will accept an
  /// AST file the was created out of an AST with compiler errors,
  /// otherwise it will reject it.
  ASTReader(Preprocessor &PP, ASTContext &Context, StringRef isysroot = "",
            bool DisableValidation = false,
            bool AllowASTWithCompilerErrors = false);

  ~ASTReader();

  SourceManager &getSourceManager() const { return SourceMgr; }

  /// \brief Flags that indicate what kind of AST loading failures the client
  /// of the AST reader can directly handle.
  ///
  /// When a client states that it can handle a particular kind of failure,
  /// the AST reader will not emit errors when producing that kind of failure.
  enum LoadFailureCapabilities {
    /// \brief The client can't handle any AST loading failures.
    ARR_None = 0,
    /// \brief The client can handle an AST file that cannot load because it
    /// is out-of-date relative to its input files.
    ARR_OutOfDate = 0x1,
    /// \brief The client can handle an AST file that cannot load because it
    /// was built with a different version of Clang.
    ARR_VersionMismatch = 0x2,
    /// \brief The client can handle an AST file that cannot load because it's
    /// compiled configuration doesn't match that of the context it was
    /// loaded into.
    ARR_ConfigurationMismatch = 0x4
  };

  /// \brief Load the AST file designated by the given file name.
  ///
  /// \param FileName The name of the AST file to load.
  ///
  /// \param Type The kind of AST being loaded, e.g., PCH, module, main file,
  /// or preamble.
  ///
  /// \param ClientLoadCapabilities The set of client load-failure
  /// capabilities, represented as a bitset of the enumerators of
  /// LoadFailureCapabilities.
  ASTReadResult ReadAST(const std::string &FileName, ModuleKind Type,
                        unsigned ClientLoadCapabilities);

  /// \brief Make the entities in the given module and any of its (non-explicit)
  /// submodules visible to name lookup.
  ///
  /// \param Mod The module whose names should be made visible.
  ///
  /// \param NameVisibility The level of visibility to give the names in the
  /// module.  Visibility can only be increased over time.
  void makeModuleVisible(Module *Mod, 
                         Module::NameVisibilityKind NameVisibility);
  
  /// \brief Make the names within this set of hidden names visible.
  void makeNamesVisible(const HiddenNames &Names);
  
  /// \brief Set the AST callbacks listener.
  void setListener(ASTReaderListener *listener) {
    Listener.reset(listener);
  }

  /// \brief Set the AST deserialization listener.
  void setDeserializationListener(ASTDeserializationListener *Listener);

  /// \brief Initializes the ASTContext
  void InitializeContext();

  /// \brief Add in-memory (virtual file) buffer.
  void addInMemoryBuffer(StringRef &FileName, llvm::MemoryBuffer *Buffer) {
    ModuleMgr.addInMemoryBuffer(FileName, Buffer);
  }

  /// \brief Finalizes the AST reader's state before writing an AST file to
  /// disk.
  ///
  /// This operation may undo temporary state in the AST that should not be
  /// emitted.
  void finalizeForWriting();

  /// \brief Retrieve the module manager.
  ModuleManager &getModuleManager() { return ModuleMgr; }

  /// \brief Retrieve the preprocessor.
  Preprocessor &getPreprocessor() const { return PP; }

  /// \brief Retrieve the name of the original source file name for the primary
  /// module file.
  StringRef getOriginalSourceFile() {
    return ModuleMgr.getPrimaryModule().OriginalSourceFileName; 
  }

  /// \brief Retrieve the name of the original source file name directly from
  /// the AST file, without actually loading the AST file.
  static std::string getOriginalSourceFile(const std::string &ASTFileName,
                                           FileManager &FileMgr,
                                           DiagnosticsEngine &Diags);

  /// \brief Read the control block for the named AST file.
  ///
  /// \returns true if an error occurred, false otherwise.
  static bool readASTFileControlBlock(StringRef Filename,
                                      FileManager &FileMgr,
                                      ASTReaderListener &Listener);

  /// \brief Determine whether the given AST file is acceptable to load into a
  /// translation unit with the given language and target options.
  static bool isAcceptableASTFile(StringRef Filename,
                                  FileManager &FileMgr,
                                  const LangOptions &LangOpts,
                                  const TargetOptions &TargetOpts,
                                  const PreprocessorOptions &PPOpts);

  /// \brief Returns the suggested contents of the predefines buffer,
  /// which contains a (typically-empty) subset of the predefines
  /// build prior to including the precompiled header.
  const std::string &getSuggestedPredefines() { return SuggestedPredefines; }

  /// \brief Read a preallocated preprocessed entity from the external source.
  ///
  /// \returns null if an error occurred that prevented the preprocessed
  /// entity from being loaded.
  virtual PreprocessedEntity *ReadPreprocessedEntity(unsigned Index);

  /// \brief Returns a pair of [Begin, End) indices of preallocated
  /// preprocessed entities that \p Range encompasses.
  virtual std::pair<unsigned, unsigned>
      findPreprocessedEntitiesInRange(SourceRange Range);

  /// \brief Optionally returns true or false if the preallocated preprocessed
  /// entity with index \p Index came from file \p FID.
  virtual llvm::Optional<bool> isPreprocessedEntityInFileID(unsigned Index,
                                                            FileID FID);

  /// \brief Read the header file information for the given file entry.
  virtual HeaderFileInfo GetHeaderFileInfo(const FileEntry *FE);

  void ReadPragmaDiagnosticMappings(DiagnosticsEngine &Diag);

  /// \brief Returns the number of source locations found in the chain.
  unsigned getTotalNumSLocs() const {
    return TotalNumSLocEntries;
  }

  /// \brief Returns the number of identifiers found in the chain.
  unsigned getTotalNumIdentifiers() const {
    return static_cast<unsigned>(IdentifiersLoaded.size());
  }

  /// \brief Returns the number of macros found in the chain.
  unsigned getTotalNumMacros() const {
    return static_cast<unsigned>(MacrosLoaded.size());
  }

  /// \brief Returns the number of types found in the chain.
  unsigned getTotalNumTypes() const {
    return static_cast<unsigned>(TypesLoaded.size());
  }

  /// \brief Returns the number of declarations found in the chain.
  unsigned getTotalNumDecls() const {
    return static_cast<unsigned>(DeclsLoaded.size());
  }

  /// \brief Returns the number of submodules known.
  unsigned getTotalNumSubmodules() const {
    return static_cast<unsigned>(SubmodulesLoaded.size());
  }
  
  /// \brief Returns the number of selectors found in the chain.
  unsigned getTotalNumSelectors() const {
    return static_cast<unsigned>(SelectorsLoaded.size());
  }

  /// \brief Returns the number of preprocessed entities known to the AST
  /// reader.
  unsigned getTotalNumPreprocessedEntities() const {
    unsigned Result = 0;
    for (ModuleConstIterator I = ModuleMgr.begin(),
        E = ModuleMgr.end(); I != E; ++I) {
      Result += (*I)->NumPreprocessedEntities;
    }

    return Result;
  }

  /// \brief Returns the number of C++ base specifiers found in the chain.
  unsigned getTotalNumCXXBaseSpecifiers() const {
    return NumCXXBaseSpecifiersLoaded;
  }

  /// \brief Reads a TemplateArgumentLocInfo appropriate for the
  /// given TemplateArgument kind.
  TemplateArgumentLocInfo
  GetTemplateArgumentLocInfo(ModuleFile &F, TemplateArgument::ArgKind Kind,
                             const RecordData &Record, unsigned &Idx);

  /// \brief Reads a TemplateArgumentLoc.
  TemplateArgumentLoc
  ReadTemplateArgumentLoc(ModuleFile &F,
                          const RecordData &Record, unsigned &Idx);

  /// \brief Reads a declarator info from the given record.
  TypeSourceInfo *GetTypeSourceInfo(ModuleFile &F,
                                    const RecordData &Record, unsigned &Idx);

  /// \brief Resolve a type ID into a type, potentially building a new
  /// type.
  QualType GetType(serialization::TypeID ID);

  /// \brief Resolve a local type ID within a given AST file into a type.
  QualType getLocalType(ModuleFile &F, unsigned LocalID);

  /// \brief Map a local type ID within a given AST file into a global type ID.
  serialization::TypeID getGlobalTypeID(ModuleFile &F, unsigned LocalID) const;

  /// \brief Read a type from the current position in the given record, which
  /// was read from the given AST file.
  QualType readType(ModuleFile &F, const RecordData &Record, unsigned &Idx) {
    if (Idx >= Record.size())
      return QualType();

    return getLocalType(F, Record[Idx++]);
  }

  /// \brief Map from a local declaration ID within a given module to a
  /// global declaration ID.
  serialization::DeclID getGlobalDeclID(ModuleFile &F,
                                      serialization::LocalDeclID LocalID) const;

  /// \brief Returns true if global DeclID \p ID originated from module \p M.
  bool isDeclIDFromModule(serialization::GlobalDeclID ID, ModuleFile &M) const;

  /// \brief Retrieve the module file that owns the given declaration, or NULL
  /// if the declaration is not from a module file.
  ModuleFile *getOwningModuleFile(Decl *D);
  
  /// \brief Returns the source location for the decl \p ID.
  SourceLocation getSourceLocationForDeclID(serialization::GlobalDeclID ID);

  /// \brief Resolve a declaration ID into a declaration, potentially
  /// building a new declaration.
  Decl *GetDecl(serialization::DeclID ID);
  virtual Decl *GetExternalDecl(uint32_t ID);

  /// \brief Reads a declaration with the given local ID in the given module.
  Decl *GetLocalDecl(ModuleFile &F, uint32_t LocalID) {
    return GetDecl(getGlobalDeclID(F, LocalID));
  }

  /// \brief Reads a declaration with the given local ID in the given module.
  ///
  /// \returns The requested declaration, casted to the given return type.
  template<typename T>
  T *GetLocalDeclAs(ModuleFile &F, uint32_t LocalID) {
    return cast_or_null<T>(GetLocalDecl(F, LocalID));
  }

  /// \brief Map a global declaration ID into the declaration ID used to 
  /// refer to this declaration within the given module fule.
  ///
  /// \returns the global ID of the given declaration as known in the given
  /// module file.
  serialization::DeclID 
  mapGlobalIDToModuleFileGlobalID(ModuleFile &M,
                                  serialization::DeclID GlobalID);
  
  /// \brief Reads a declaration ID from the given position in a record in the
  /// given module.
  ///
  /// \returns The declaration ID read from the record, adjusted to a global ID.
  serialization::DeclID ReadDeclID(ModuleFile &F, const RecordData &Record,
                                   unsigned &Idx);

  /// \brief Reads a declaration from the given position in a record in the
  /// given module.
  Decl *ReadDecl(ModuleFile &F, const RecordData &R, unsigned &I) {
    return GetDecl(ReadDeclID(F, R, I));
  }

  /// \brief Reads a declaration from the given position in a record in the
  /// given module.
  ///
  /// \returns The declaration read from this location, casted to the given
  /// result type.
  template<typename T>
  T *ReadDeclAs(ModuleFile &F, const RecordData &R, unsigned &I) {
    return cast_or_null<T>(GetDecl(ReadDeclID(F, R, I)));
  }

  /// \brief Read a CXXBaseSpecifiers ID form the given record and
  /// return its global bit offset.
  uint64_t readCXXBaseSpecifiers(ModuleFile &M, const RecordData &Record,
                                 unsigned &Idx);

  virtual CXXBaseSpecifier *GetExternalCXXBaseSpecifiers(uint64_t Offset);

  /// \brief Resolve the offset of a statement into a statement.
  ///
  /// This operation will read a new statement from the external
  /// source each time it is called, and is meant to be used via a
  /// LazyOffsetPtr (which is used by Decls for the body of functions, etc).
  virtual Stmt *GetExternalDeclStmt(uint64_t Offset);

  /// ReadBlockAbbrevs - Enter a subblock of the specified BlockID with the
  /// specified cursor.  Read the abbreviations that are at the top of the block
  /// and then leave the cursor pointing into the block.
  bool ReadBlockAbbrevs(llvm::BitstreamCursor &Cursor, unsigned BlockID);

  /// \brief Finds all the visible declarations with a given name.
  /// The current implementation of this method just loads the entire
  /// lookup table as unmaterialized references.
  virtual DeclContext::lookup_result
  FindExternalVisibleDeclsByName(const DeclContext *DC,
                                 DeclarationName Name);

  /// \brief Read all of the declarations lexically stored in a
  /// declaration context.
  ///
  /// \param DC The declaration context whose declarations will be
  /// read.
  ///
  /// \param Decls Vector that will contain the declarations loaded
  /// from the external source. The caller is responsible for merging
  /// these declarations with any declarations already stored in the
  /// declaration context.
  ///
  /// \returns true if there was an error while reading the
  /// declarations for this declaration context.
  virtual ExternalLoadResult FindExternalLexicalDecls(const DeclContext *DC,
                                        bool (*isKindWeWant)(Decl::Kind),
                                        SmallVectorImpl<Decl*> &Decls);

  /// \brief Get the decls that are contained in a file in the Offset/Length
  /// range. \p Length can be 0 to indicate a point at \p Offset instead of
  /// a range.
  virtual void FindFileRegionDecls(FileID File, unsigned Offset,unsigned Length,
                                   SmallVectorImpl<Decl *> &Decls);

  /// \brief Notify ASTReader that we started deserialization of
  /// a decl or type so until FinishedDeserializing is called there may be
  /// decls that are initializing. Must be paired with FinishedDeserializing.
  virtual void StartedDeserializing() { ++NumCurrentElementsDeserializing; }

  /// \brief Notify ASTReader that we finished the deserialization of
  /// a decl or type. Must be paired with StartedDeserializing.
  virtual void FinishedDeserializing();

  /// \brief Function that will be invoked when we begin parsing a new
  /// translation unit involving this external AST source.
  ///
  /// This function will provide all of the external definitions to
  /// the ASTConsumer.
  virtual void StartTranslationUnit(ASTConsumer *Consumer);

  /// \brief Print some statistics about AST usage.
  virtual void PrintStats();

  /// \brief Dump information about the AST reader to standard error.
  void dump();

  /// Return the amount of memory used by memory buffers, breaking down
  /// by heap-backed versus mmap'ed memory.
  virtual void getMemoryBufferSizes(MemoryBufferSizes &sizes) const;

  /// \brief Initialize the semantic source with the Sema instance
  /// being used to perform semantic analysis on the abstract syntax
  /// tree.
  virtual void InitializeSema(Sema &S);

  /// \brief Inform the semantic consumer that Sema is no longer available.
  virtual void ForgetSema() { SemaObj = 0; }

  /// \brief Retrieve the IdentifierInfo for the named identifier.
  ///
  /// This routine builds a new IdentifierInfo for the given identifier. If any
  /// declarations with this name are visible from translation unit scope, their
  /// declarations will be deserialized and introduced into the declaration
  /// chain of the identifier.
  virtual IdentifierInfo *get(const char *NameStart, const char *NameEnd);
  IdentifierInfo *get(StringRef Name) {
    return get(Name.begin(), Name.end());
  }

  /// \brief Retrieve an iterator into the set of all identifiers
  /// in all loaded AST files.
  virtual IdentifierIterator *getIdentifiers() const;

  /// \brief Load the contents of the global method pool for a given
  /// selector.
  virtual void ReadMethodPool(Selector Sel);

  /// \brief Load the set of namespaces that are known to the external source,
  /// which will be used during typo correction.
  virtual void ReadKnownNamespaces(
                           SmallVectorImpl<NamespaceDecl *> &Namespaces);

  virtual void ReadTentativeDefinitions(
                 SmallVectorImpl<VarDecl *> &TentativeDefs);

  virtual void ReadUnusedFileScopedDecls(
                 SmallVectorImpl<const DeclaratorDecl *> &Decls);

  virtual void ReadDelegatingConstructors(
                 SmallVectorImpl<CXXConstructorDecl *> &Decls);

  virtual void ReadExtVectorDecls(SmallVectorImpl<TypedefNameDecl *> &Decls);

  virtual void ReadDynamicClasses(SmallVectorImpl<CXXRecordDecl *> &Decls);

  virtual void ReadLocallyScopedExternalDecls(
                 SmallVectorImpl<NamedDecl *> &Decls);

  virtual void ReadReferencedSelectors(
                 SmallVectorImpl<std::pair<Selector, SourceLocation> > &Sels);

  virtual void ReadWeakUndeclaredIdentifiers(
                 SmallVectorImpl<std::pair<IdentifierInfo *, WeakInfo> > &WI);

  virtual void ReadUsedVTables(SmallVectorImpl<ExternalVTableUse> &VTables);

  virtual void ReadPendingInstantiations(
                 SmallVectorImpl<std::pair<ValueDecl *,
                                           SourceLocation> > &Pending);

  /// \brief Load a selector from disk, registering its ID if it exists.
  void LoadSelector(Selector Sel);

  void SetIdentifierInfo(unsigned ID, IdentifierInfo *II);
  void SetGloballyVisibleDecls(IdentifierInfo *II,
                               const SmallVectorImpl<uint32_t> &DeclIDs,
                               bool Nonrecursive = false);

  /// \brief Report a diagnostic.
  DiagnosticBuilder Diag(unsigned DiagID);

  /// \brief Report a diagnostic.
  DiagnosticBuilder Diag(SourceLocation Loc, unsigned DiagID);

  IdentifierInfo *DecodeIdentifierInfo(serialization::IdentifierID ID);

  IdentifierInfo *GetIdentifierInfo(ModuleFile &M, const RecordData &Record,
                                    unsigned &Idx) {
    return DecodeIdentifierInfo(getGlobalIdentifierID(M, Record[Idx++]));
  }

  virtual IdentifierInfo *GetIdentifier(serialization::IdentifierID ID) {
    // Note that we are loading an identifier.
    Deserializing AnIdentifier(this);

    return DecodeIdentifierInfo(ID);
  }

  IdentifierInfo *getLocalIdentifier(ModuleFile &M, unsigned LocalID);

  serialization::IdentifierID getGlobalIdentifierID(ModuleFile &M,
                                                    unsigned LocalID);

  /// \brief Retrieve the macro with the given ID.
  MacroInfo *getMacro(serialization::MacroID ID, MacroInfo *Hint = 0);

  /// \brief Retrieve the global macro ID corresponding to the given local
  /// ID within the given module file.
  serialization::MacroID getGlobalMacroID(ModuleFile &M, unsigned LocalID);

  /// \brief Read the source location entry with index ID.
  virtual bool ReadSLocEntry(int ID);

  /// \brief Retrieve the global submodule ID given a module and its local ID
  /// number.
  serialization::SubmoduleID 
  getGlobalSubmoduleID(ModuleFile &M, unsigned LocalID);
  
  /// \brief Retrieve the submodule that corresponds to a global submodule ID.
  ///
  Module *getSubmodule(serialization::SubmoduleID GlobalID);
  
  /// \brief Retrieve a selector from the given module with its local ID
  /// number.
  Selector getLocalSelector(ModuleFile &M, unsigned LocalID);

  Selector DecodeSelector(serialization::SelectorID Idx);

  virtual Selector GetExternalSelector(serialization::SelectorID ID);
  uint32_t GetNumExternalSelectors();

  Selector ReadSelector(ModuleFile &M, const RecordData &Record, unsigned &Idx) {
    return getLocalSelector(M, Record[Idx++]);
  }

  /// \brief Retrieve the global selector ID that corresponds to this
  /// the local selector ID in a given module.
  serialization::SelectorID getGlobalSelectorID(ModuleFile &F,
                                                unsigned LocalID) const;

  /// \brief Read a declaration name.
  DeclarationName ReadDeclarationName(ModuleFile &F,
                                      const RecordData &Record, unsigned &Idx);
  void ReadDeclarationNameLoc(ModuleFile &F,
                              DeclarationNameLoc &DNLoc, DeclarationName Name,
                              const RecordData &Record, unsigned &Idx);
  void ReadDeclarationNameInfo(ModuleFile &F, DeclarationNameInfo &NameInfo,
                               const RecordData &Record, unsigned &Idx);

  void ReadQualifierInfo(ModuleFile &F, QualifierInfo &Info,
                         const RecordData &Record, unsigned &Idx);

  NestedNameSpecifier *ReadNestedNameSpecifier(ModuleFile &F,
                                               const RecordData &Record,
                                               unsigned &Idx);

  NestedNameSpecifierLoc ReadNestedNameSpecifierLoc(ModuleFile &F,
                                                    const RecordData &Record,
                                                    unsigned &Idx);

  /// \brief Read a template name.
  TemplateName ReadTemplateName(ModuleFile &F, const RecordData &Record,
                                unsigned &Idx);

  /// \brief Read a template argument.
  TemplateArgument ReadTemplateArgument(ModuleFile &F,
                                        const RecordData &Record,unsigned &Idx);

  /// \brief Read a template parameter list.
  TemplateParameterList *ReadTemplateParameterList(ModuleFile &F,
                                                   const RecordData &Record,
                                                   unsigned &Idx);

  /// \brief Read a template argument array.
  void
  ReadTemplateArgumentList(SmallVector<TemplateArgument, 8> &TemplArgs,
                           ModuleFile &F, const RecordData &Record,
                           unsigned &Idx);

  /// \brief Read a UnresolvedSet structure.
  void ReadUnresolvedSet(ModuleFile &F, UnresolvedSetImpl &Set,
                         const RecordData &Record, unsigned &Idx);

  /// \brief Read a C++ base specifier.
  CXXBaseSpecifier ReadCXXBaseSpecifier(ModuleFile &F,
                                        const RecordData &Record,unsigned &Idx);

  /// \brief Read a CXXCtorInitializer array.
  std::pair<CXXCtorInitializer **, unsigned>
  ReadCXXCtorInitializers(ModuleFile &F, const RecordData &Record,
                          unsigned &Idx);

  /// \brief Read a source location from raw form.
  SourceLocation ReadSourceLocation(ModuleFile &ModuleFile, unsigned Raw) const {
    SourceLocation Loc = SourceLocation::getFromRawEncoding(Raw);
    assert(ModuleFile.SLocRemap.find(Loc.getOffset()) != ModuleFile.SLocRemap.end() &&
           "Cannot find offset to remap.");
    int Remap = ModuleFile.SLocRemap.find(Loc.getOffset())->second;
    return Loc.getLocWithOffset(Remap);
  }

  /// \brief Read a source location.
  SourceLocation ReadSourceLocation(ModuleFile &ModuleFile,
                                    const RecordData &Record, unsigned& Idx) {
    return ReadSourceLocation(ModuleFile, Record[Idx++]);
  }

  /// \brief Read a source range.
  SourceRange ReadSourceRange(ModuleFile &F,
                              const RecordData &Record, unsigned& Idx);

  /// \brief Read an integral value
  llvm::APInt ReadAPInt(const RecordData &Record, unsigned &Idx);

  /// \brief Read a signed integral value
  llvm::APSInt ReadAPSInt(const RecordData &Record, unsigned &Idx);

  /// \brief Read a floating-point value
  llvm::APFloat ReadAPFloat(const RecordData &Record, unsigned &Idx);

  // \brief Read a string
  static std::string ReadString(const RecordData &Record, unsigned &Idx);

  /// \brief Read a version tuple.
  static VersionTuple ReadVersionTuple(const RecordData &Record, unsigned &Idx);

  CXXTemporary *ReadCXXTemporary(ModuleFile &F, const RecordData &Record,
                                 unsigned &Idx);

  /// \brief Reads attributes from the current stream position.
  void ReadAttributes(ModuleFile &F, AttrVec &Attrs,
                      const RecordData &Record, unsigned &Idx);

  /// \brief Reads a statement.
  Stmt *ReadStmt(ModuleFile &F);

  /// \brief Reads an expression.
  Expr *ReadExpr(ModuleFile &F);

  /// \brief Reads a sub-statement operand during statement reading.
  Stmt *ReadSubStmt() {
    assert(ReadingKind == Read_Stmt &&
           "Should be called only during statement reading!");
    // Subexpressions are stored from last to first, so the next Stmt we need
    // is at the back of the stack.
    assert(!StmtStack.empty() && "Read too many sub statements!");
    return StmtStack.pop_back_val();
  }

  /// \brief Reads a sub-expression operand during statement reading.
  Expr *ReadSubExpr();

  /// \brief Reads the macro record located at the given offset.
  void ReadMacroRecord(ModuleFile &F, uint64_t Offset, MacroInfo *Hint = 0);

  /// \brief Determine the global preprocessed entity ID that corresponds to
  /// the given local ID within the given module.
  serialization::PreprocessedEntityID
  getGlobalPreprocessedEntityID(ModuleFile &M, unsigned LocalID) const;

  /// \brief Note that the identifier has a macro history.
  ///
  /// \param II The name of the macro.
  ///
  /// \param IDs The global macro IDs that are associated with this identifier.
  void setIdentifierIsMacro(IdentifierInfo *II,
                            ArrayRef<serialization::MacroID> IDs);

  /// \brief Read the set of macros defined by this external macro source.
  virtual void ReadDefinedMacros();

  /// \brief Update an out-of-date identifier.
  virtual void updateOutOfDateIdentifier(IdentifierInfo &II);

  /// \brief Note that this identifier is up-to-date.
  void markIdentifierUpToDate(IdentifierInfo *II);

  /// \brief Load all external visible decls in the given DeclContext.
  void completeVisibleDeclsMap(const DeclContext *DC);

  /// \brief Retrieve the AST context that this AST reader supplements.
  ASTContext &getContext() { return Context; }

  // \brief Contains declarations that were loaded before we have
  // access to a Sema object.
  SmallVector<NamedDecl *, 16> PreloadedDecls;

  /// \brief Retrieve the semantic analysis object used to analyze the
  /// translation unit in which the precompiled header is being
  /// imported.
  Sema *getSema() { return SemaObj; }

  /// \brief Retrieve the identifier table associated with the
  /// preprocessor.
  IdentifierTable &getIdentifierTable();

  /// \brief Record that the given ID maps to the given switch-case
  /// statement.
  void RecordSwitchCaseID(SwitchCase *SC, unsigned ID);

  /// \brief Retrieve the switch-case statement with the given ID.
  SwitchCase *getSwitchCaseWithID(unsigned ID);

  void ClearSwitchCaseIDs();

  /// \brief Cursors for comments blocks.
  SmallVector<std::pair<llvm::BitstreamCursor,
                        serialization::ModuleFile *>, 8> CommentsCursors;

  /// \brief Loads comments ranges.
  void ReadComments();
};

/// \brief Helper class that saves the current stream position and
/// then restores it when destroyed.
struct SavedStreamPosition {
  explicit SavedStreamPosition(llvm::BitstreamCursor &Cursor)
  : Cursor(Cursor), Offset(Cursor.GetCurrentBitNo()) { }

  ~SavedStreamPosition() {
    Cursor.JumpToBit(Offset);
  }

private:
  llvm::BitstreamCursor &Cursor;
  uint64_t Offset;
};

inline void PCHValidator::Error(const char *Msg) {
  Reader.Error(Msg);
}

} // end namespace clang

#endif
