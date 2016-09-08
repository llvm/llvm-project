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

#ifndef LLVM_CLANG_SERIALIZATION_ASTREADER_H
#define LLVM_CLANG_SERIALIZATION_ASTREADER_H

#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/TemplateBase.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/Version.h"
#include "clang/Lex/ExternalPreprocessorSource.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/PreprocessingRecord.h"
#include "clang/Sema/ExternalSemaSource.h"
#include "clang/Sema/IdentifierResolver.h"
#include "clang/Serialization/ASTBitCodes.h"
#include "clang/Serialization/ContinuousRangeMap.h"
#include "clang/Serialization/Module.h"
#include "clang/Serialization/ModuleFileExtension.h"
#include "clang/Serialization/ModuleManager.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Timer.h"
#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace llvm {
  class MemoryBuffer;
  class APInt;
  class APSInt;
  class APFloat;
}

namespace clang {

class SourceManager;
class HeaderSearchOptions;
class FileManager;
class AddrLabelExpr;
class ASTConsumer;
class ASTContext;
class ASTIdentifierIterator;
class ASTUnit; // FIXME: Layering violation and egregious hack.
class Attr;
class Decl;
class DeclContext;
class DefMacroDirective;
class DiagnosticOptions;
class NestedNameSpecifier;
class CXXBaseSpecifier;
class CXXConstructorDecl;
class CXXCtorInitializer;
class GlobalModuleIndex;
class GotoStmt;
class MacroDefinition;
class MacroDirective;
class ModuleMacro;
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
class LazyASTUnresolvedSet;

/// \brief Abstract interface for callback invocations by the ASTReader.
///
/// While reading an AST file, the ASTReader will call the methods of the
/// listener to pass on specific information. Some of the listener methods can
/// return true to indicate to the ASTReader that the information (and
/// consequently the AST file) is invalid.
class ASTReaderListener {
public:
  virtual ~ASTReaderListener();

  /// \brief Receives the full Clang version information.
  ///
  /// \returns true to indicate that the version is invalid. Subclasses should
  /// generally defer to this implementation.
  virtual bool ReadFullVersionInformation(StringRef FullVersion) {
    return FullVersion != getClangFullRepositoryVersion();
  }

  virtual void ReadModuleName(StringRef ModuleName) {}
  virtual void ReadModuleMapFile(StringRef ModuleMapPath) {}

  /// \brief Receives the language options.
  ///
  /// \returns true to indicate the options are invalid or false otherwise.
  virtual bool ReadLanguageOptions(const LangOptions &LangOpts,
                                   bool Complain,
                                   bool AllowCompatibleDifferences) {
    return false;
  }

  /// \brief Receives the target options.
  ///
  /// \returns true to indicate the target options are invalid, or false
  /// otherwise.
  virtual bool ReadTargetOptions(const TargetOptions &TargetOpts, bool Complain,
                                 bool AllowCompatibleDifferences) {
    return false;
  }

  /// \brief Receives the diagnostic options.
  ///
  /// \returns true to indicate the diagnostic options are invalid, or false
  /// otherwise.
  virtual bool
  ReadDiagnosticOptions(IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts,
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
                                       StringRef SpecificModuleCachePath,
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

  /// \brief Receives __COUNTER__ value.
  virtual void ReadCounter(const serialization::ModuleFile &M,
                           unsigned Value) {}

  /// This is called for each AST file loaded.
  virtual void visitModuleFile(StringRef Filename,
                               serialization::ModuleKind Kind) {}

  /// \brief Returns true if this \c ASTReaderListener wants to receive the
  /// input files of the AST file via \c visitInputFile, false otherwise.
  virtual bool needsInputFileVisitation() { return false; }
  /// \brief Returns true if this \c ASTReaderListener wants to receive the
  /// system input files of the AST file via \c visitInputFile, false otherwise.
  virtual bool needsSystemInputFileVisitation() { return false; }
  /// \brief if \c needsInputFileVisitation returns true, this is called for
  /// each non-system input file of the AST File. If
  /// \c needsSystemInputFileVisitation is true, then it is called for all
  /// system input files as well.
  ///
  /// \returns true to continue receiving the next input file, false to stop.
  virtual bool visitInputFile(StringRef Filename, bool isSystem,
                              bool isOverridden, bool isExplicitModule) {
    return true;
  }

  /// \brief Returns true if this \c ASTReaderListener wants to receive the
  /// imports of the AST file via \c visitImport, false otherwise.
  virtual bool needsImportVisitation() const { return false; }
  /// \brief If needsImportVisitation returns \c true, this is called for each
  /// AST file imported by this AST file.
  virtual void visitImport(StringRef Filename) {}

  /// Indicates that a particular module file extension has been read.
  virtual void readModuleFileExtension(
                 const ModuleFileExtensionMetadata &Metadata) {}
};

/// \brief Simple wrapper class for chaining listeners.
class ChainedASTReaderListener : public ASTReaderListener {
  std::unique_ptr<ASTReaderListener> First;
  std::unique_ptr<ASTReaderListener> Second;

public:
  /// Takes ownership of \p First and \p Second.
  ChainedASTReaderListener(std::unique_ptr<ASTReaderListener> First,
                           std::unique_ptr<ASTReaderListener> Second)
      : First(std::move(First)), Second(std::move(Second)) {}

  std::unique_ptr<ASTReaderListener> takeFirst() { return std::move(First); }
  std::unique_ptr<ASTReaderListener> takeSecond() { return std::move(Second); }

  bool ReadFullVersionInformation(StringRef FullVersion) override;
  void ReadModuleName(StringRef ModuleName) override;
  void ReadModuleMapFile(StringRef ModuleMapPath) override;
  bool ReadLanguageOptions(const LangOptions &LangOpts, bool Complain,
                           bool AllowCompatibleDifferences) override;
  bool ReadTargetOptions(const TargetOptions &TargetOpts, bool Complain,
                         bool AllowCompatibleDifferences) override;
  bool ReadDiagnosticOptions(IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts,
                             bool Complain) override;
  bool ReadFileSystemOptions(const FileSystemOptions &FSOpts,
                             bool Complain) override;

  bool ReadHeaderSearchOptions(const HeaderSearchOptions &HSOpts,
                               StringRef SpecificModuleCachePath,
                               bool Complain) override;
  bool ReadPreprocessorOptions(const PreprocessorOptions &PPOpts,
                               bool Complain,
                               std::string &SuggestedPredefines) override;

  void ReadCounter(const serialization::ModuleFile &M, unsigned Value) override;
  bool needsInputFileVisitation() override;
  bool needsSystemInputFileVisitation() override;
  void visitModuleFile(StringRef Filename,
                       serialization::ModuleKind Kind) override;
  bool visitInputFile(StringRef Filename, bool isSystem,
                      bool isOverridden, bool isExplicitModule) override;
  void readModuleFileExtension(
         const ModuleFileExtensionMetadata &Metadata) override;
};

/// \brief ASTReaderListener implementation to validate the information of
/// the PCH file against an initialized Preprocessor.
class PCHValidator : public ASTReaderListener {
  Preprocessor &PP;
  ASTReader &Reader;

public:
  PCHValidator(Preprocessor &PP, ASTReader &Reader)
    : PP(PP), Reader(Reader) {}

  bool ReadLanguageOptions(const LangOptions &LangOpts, bool Complain,
                           bool AllowCompatibleDifferences) override;
  bool ReadTargetOptions(const TargetOptions &TargetOpts, bool Complain,
                         bool AllowCompatibleDifferences) override;
  bool ReadDiagnosticOptions(IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts,
                             bool Complain) override;
  bool ReadPreprocessorOptions(const PreprocessorOptions &PPOpts, bool Complain,
                               std::string &SuggestedPredefines) override;
  bool ReadHeaderSearchOptions(const HeaderSearchOptions &HSOpts,
                               StringRef SpecificModuleCachePath,
                               bool Complain) override;
  void ReadCounter(const serialization::ModuleFile &M, unsigned Value) override;

private:
  void Error(const char *Msg);
};

/// \brief ASTReaderListenter implementation to set SuggestedPredefines of
/// ASTReader which is required to use a pch file. This is the replacement
/// of PCHValidator or SimplePCHValidator when using a pch file without
/// validating it.
class SimpleASTReaderListener : public ASTReaderListener {
  Preprocessor &PP;

public:
  SimpleASTReaderListener(Preprocessor &PP)
    : PP(PP) {}

  bool ReadPreprocessorOptions(const PreprocessorOptions &PPOpts, bool Complain,
                               std::string &SuggestedPredefines) override;
};

namespace serialization {

class ReadMethodPoolVisitor;

namespace reader {
  class ASTIdentifierLookupTrait;
  /// \brief The on-disk hash table(s) used for DeclContext name lookup.
  struct DeclContextLookupTable;
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
    public ExternalSLocEntrySource
{
public:
  typedef SmallVector<uint64_t, 64> RecordData;
  typedef SmallVectorImpl<uint64_t> RecordDataImpl;

  /// \brief The result of reading the control block of an AST file, which
  /// can fail for various reasons.
  enum ASTReadResult {
    /// \brief The control block was read successfully. Aside from failures,
    /// the AST file is safe to read into the current context.
    Success,
    /// \brief The AST file itself appears corrupted.
    Failure,
    /// \brief The AST file was missing.
    Missing,
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
  std::unique_ptr<ASTReaderListener> Listener;

  /// \brief The receiver of deserialization events.
  ASTDeserializationListener *DeserializationListener;
  bool OwnsDeserializationListener;

  SourceManager &SourceMgr;
  FileManager &FileMgr;
  const PCHContainerReader &PCHContainerRdr;
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

  /// \brief A dummy identifier resolver used to merge TU-scope declarations in
  /// C, for the cases where we don't have a Sema object to provide a real
  /// identifier resolver.
  IdentifierResolver DummyIdResolver;

  /// A mapping from extension block names to module file extensions.
  llvm::StringMap<IntrusiveRefCntPtr<ModuleFileExtension>> ModuleFileExtensions;

  /// \brief A timer used to track the time spent deserializing.
  std::unique_ptr<llvm::Timer> ReadTimer;

  /// \brief The location where the module file will be considered as
  /// imported from. For non-module AST types it should be invalid.
  SourceLocation CurrentImportLoc;

  /// \brief The global module index, if loaded.
  std::unique_ptr<GlobalModuleIndex> GlobalIndex;

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

  /// \brief Declaration updates for already-loaded declarations that we need
  /// to apply once we finish processing an import.
  llvm::SmallVector<std::pair<serialization::GlobalDeclID, Decl*>, 16>
      PendingUpdateRecords;

  enum class PendingFakeDefinitionKind { NotFake, Fake, FakeLoaded };

  /// \brief The DefinitionData pointers that we faked up for class definitions
  /// that we needed but hadn't loaded yet.
  llvm::DenseMap<void *, PendingFakeDefinitionKind> PendingFakeDefinitionData;

  /// \brief Exception specification updates that have been loaded but not yet
  /// propagated across the relevant redeclaration chain. The map key is the
  /// canonical declaration (used only for deduplication) and the value is a
  /// declaration that has an exception specification.
  llvm::SmallMapVector<Decl *, FunctionDecl *, 4> PendingExceptionSpecUpdates;

  /// \brief Declarations that have been imported and have typedef names for
  /// linkage purposes.
  llvm::DenseMap<std::pair<DeclContext*, IdentifierInfo*>, NamedDecl*>
      ImportedTypedefNamesForLinkage;

  /// \brief Mergeable declaration contexts that have anonymous declarations
  /// within them, and those anonymous declarations.
  llvm::DenseMap<DeclContext*, llvm::SmallVector<NamedDecl*, 2>>
    AnonymousDeclarationsForMerging;

  struct FileDeclsInfo {
    ModuleFile *Mod;
    ArrayRef<serialization::LocalDeclID> Decls;

    FileDeclsInfo() : Mod(nullptr) {}
    FileDeclsInfo(ModuleFile *Mod, ArrayRef<serialization::LocalDeclID> Decls)
      : Mod(Mod), Decls(Decls) {}
  };

  /// \brief Map from a FileID to the file-level declarations that it contains.
  llvm::DenseMap<FileID, FileDeclsInfo> FileDeclIDs;

  /// \brief An array of lexical contents of a declaration context, as a sequence of
  /// Decl::Kind, DeclID pairs.
  typedef ArrayRef<llvm::support::unaligned_uint32_t> LexicalContents;

  /// \brief Map from a DeclContext to its lexical contents.
  llvm::DenseMap<const DeclContext*, std::pair<ModuleFile*, LexicalContents>>
      LexicalDecls;

  /// \brief Map from the TU to its lexical contents from each module file.
  std::vector<std::pair<ModuleFile*, LexicalContents>> TULexicalDecls;

  /// \brief Map from a DeclContext to its lookup tables.
  llvm::DenseMap<const DeclContext *,
                 serialization::reader::DeclContextLookupTable> Lookups;

  // Updates for visible decls can occur for other contexts than just the
  // TU, and when we read those update records, the actual context may not
  // be available yet, so have this pending map using the ID as a key. It
  // will be realized when the context is actually loaded.
  struct PendingVisibleUpdate {
    ModuleFile *Mod;
    const unsigned char *Data;
  };
  typedef SmallVector<PendingVisibleUpdate, 1> DeclContextVisibleUpdates;

  /// \brief Updates to the visible declarations of declaration contexts that
  /// haven't been loaded yet.
  llvm::DenseMap<serialization::DeclID, DeclContextVisibleUpdates>
      PendingVisibleUpdates;

  /// \brief The set of C++ or Objective-C classes that have forward 
  /// declarations that have not yet been linked to their definitions.
  llvm::SmallPtrSet<Decl *, 4> PendingDefinitions;

  typedef llvm::MapVector<Decl *, uint64_t,
                          llvm::SmallDenseMap<Decl *, unsigned, 4>,
                          SmallVector<std::pair<Decl *, uint64_t>, 4> >
    PendingBodiesMap;

  /// \brief Functions or methods that have bodies that will be attached.
  PendingBodiesMap PendingBodies;

  /// \brief Definitions for which we have added merged definitions but not yet
  /// performed deduplication.
  llvm::SetVector<NamedDecl*> PendingMergedDefinitionsToDeduplicate;

  /// \brief Read the record that describes the lexical contents of a DC.
  bool ReadLexicalDeclContextStorage(ModuleFile &M,
                                     llvm::BitstreamCursor &Cursor,
                                     uint64_t Offset, DeclContext *DC);
  /// \brief Read the record that describes the visible contents of a DC.
  bool ReadVisibleDeclContextStorage(ModuleFile &M,
                                     llvm::BitstreamCursor &Cursor,
                                     uint64_t Offset, serialization::DeclID ID);

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

  typedef std::pair<IdentifierInfo *, serialization::SubmoduleID>
      LoadedMacroInfo;

  /// \brief A set of #undef directives that we have loaded; used to
  /// deduplicate the same #undef information coming from multiple module
  /// files.
  llvm::DenseSet<LoadedMacroInfo> LoadedUndefs;

  typedef ContinuousRangeMap<serialization::MacroID, ModuleFile *, 4>
    GlobalMacroMapType;

  /// \brief Mapping from global macro IDs to the module in which the
  /// macro resides along with the offset that should be added to the
  /// global macro ID to produce a local ID.
  GlobalMacroMapType GlobalMacroMap;

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

  /// \brief A set of hidden declarations.
  typedef SmallVector<Decl*, 2> HiddenNames;
  typedef llvm::DenseMap<Module *, HiddenNames> HiddenNamesMapType;

  /// \brief A mapping from each of the hidden submodules to the deserialized
  /// declarations in that submodule that could be made visible.
  HiddenNamesMapType HiddenNamesMap;
  
  
  /// \brief A module import, export, or conflict that hasn't yet been resolved.
  struct UnresolvedModuleRef {
    /// \brief The file in which this module resides.
    ModuleFile *File;
    
    /// \brief The module that is importing or exporting.
    Module *Mod;

    /// \brief The kind of module reference.
    enum { Import, Export, Conflict } Kind;

    /// \brief The local ID of the module that is being exported.
    unsigned ID;

    /// \brief Whether this is a wildcard export.
    unsigned IsWildcard : 1;

    /// \brief String data.
    StringRef String;
  };
  
  /// \brief The set of module imports and exports that still need to be 
  /// resolved.
  SmallVector<UnresolvedModuleRef, 2> UnresolvedModuleRefs;
  
  /// \brief A vector containing selectors that have already been loaded.
  ///
  /// This vector is indexed by the Selector ID (-1). NULL selector
  /// entries indicate that the particular selector ID has not yet
  /// been loaded.
  SmallVector<Selector, 16> SelectorsLoaded;

  typedef ContinuousRangeMap<serialization::SelectorID, ModuleFile *, 4>
    GlobalSelectorMapType;

  /// \brief Mapping from global selector IDs to the module in which the

  /// global selector ID to produce a local ID.
  GlobalSelectorMapType GlobalSelectorMap;

  /// \brief The generation number of the last time we loaded data from the
  /// global method pool for this selector.
  llvm::DenseMap<Selector, unsigned> SelectorGeneration;

  /// Whether a selector is out of date. We mark a selector as out of date
  /// if we load another module after the method pool entry was pulled in.
  llvm::DenseMap<Selector, bool> SelectorOutOfDate;

  struct PendingMacroInfo {
    ModuleFile *M;
    uint64_t MacroDirectivesOffset;

    PendingMacroInfo(ModuleFile *M, uint64_t MacroDirectivesOffset)
        : M(M), MacroDirectivesOffset(MacroDirectivesOffset) {}
  };

  typedef llvm::MapVector<IdentifierInfo *, SmallVector<PendingMacroInfo, 2> >
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
  /// This contains the data loaded from all EAGERLY_DESERIALIZED_DECLS blocks
  /// in the chain. The referenced declarations are deserialized and passed to
  /// the consumer eagerly.
  SmallVector<uint64_t, 16> EagerlyDeserializedDecls;

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

  /// \brief The IDs of all potentially unused typedef names in the chain.
  ///
  /// Sema tracks these to emit warnings.
  SmallVector<uint64_t, 16> UnusedLocalTypedefNameCandidates;

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

  /// \brief The pragma clang optimize location (if the pragma state is "off").
  SourceLocation OptimizeOffPragmaLocation;

  /// \brief The PragmaMSStructKind pragma ms_struct state if set, or -1.
  int PragmaMSStructState;

  /// \brief The PragmaMSPointersToMembersKind pragma pointers_to_members state.
  int PragmaMSPointersToMembersState;
  SourceLocation PointersToMembersPragmaLocation;

  /// \brief The OpenCL extension settings.
  SmallVector<uint64_t, 1> OpenCLExtensions;

  /// \brief A list of the namespaces we've seen.
  SmallVector<uint64_t, 4> KnownNamespaces;

  /// \brief A list of undefined decls with internal linkage followed by the
  /// SourceLocation of a matching ODR-use.
  SmallVector<uint64_t, 8> UndefinedButUsed;

  /// \brief Delete expressions to analyze at the end of translation unit.
  SmallVector<uint64_t, 8> DelayedDeleteExprs;

  // \brief A list of late parsed template function data.
  SmallVector<uint64_t, 1> LateParsedTemplates;

  struct ImportedSubmodule {
    serialization::SubmoduleID ID;
    SourceLocation ImportLoc;

    ImportedSubmodule(serialization::SubmoduleID ID, SourceLocation ImportLoc)
      : ID(ID), ImportLoc(ImportLoc) {}
  };

  /// \brief A list of modules that were imported by precompiled headers or
  /// any other non-module AST file.
  SmallVector<ImportedSubmodule, 2> ImportedModules;
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

  /// \brief Whether to accept an AST file that has a different configuration
  /// from the current compiler instance.
  bool AllowConfigurationMismatch;

  /// \brief Whether validate system input files.
  bool ValidateSystemInputs;

  /// \brief Whether we are allowed to use the global module index.
  bool UseGlobalIndex;

  /// \brief Whether we have tried loading the global module index yet.
  bool TriedLoadingGlobalIndex;

  ///\brief Whether we are currently processing update records.
  bool ProcessingUpdateRecords;

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

  /// \brief The number of lookups into identifier tables.
  unsigned NumIdentifierLookups;

  /// \brief The number of lookups into identifier tables that succeed.
  unsigned NumIdentifierLookupHits;

  /// \brief The number of selectors that have been read.
  unsigned NumSelectorsRead;

  /// \brief The number of method pool entries that have been read.
  unsigned NumMethodPoolEntriesRead;

  /// \brief The number of times we have looked up a selector in the method
  /// pool.
  unsigned NumMethodPoolLookups;

  /// \brief The number of times we have looked up a selector in the method
  /// pool and found something.
  unsigned NumMethodPoolHits;

  /// \brief The number of times we have looked up a selector in the method
  /// pool within a specific module.
  unsigned NumMethodPoolTableLookups;

  /// \brief The number of times we have looked up a selector in the method
  /// pool within a specific module and found something.
  unsigned NumMethodPoolTableHits;

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

  /// \brief The set of identifiers that were read while the AST reader was
  /// (recursively) loading declarations.
  ///
  /// The declarations on the identifier chain for these identifiers will be
  /// loaded once the recursive loading has completed.
  llvm::MapVector<IdentifierInfo *, SmallVector<uint32_t, 4> >
    PendingIdentifierInfos;

  /// \brief The set of lookup results that we have faked in order to support
  /// merging of partially deserialized decls but that we have not yet removed.
  llvm::SmallMapVector<IdentifierInfo *, SmallVector<NamedDecl*, 2>, 16>
    PendingFakeLookupResults;

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

  /// \brief The list of redeclaration chains that still need to be 
  /// reconstructed, and the local offset to the corresponding list
  /// of redeclarations.
  SmallVector<std::pair<Decl *, uint64_t>, 16> PendingDeclChains;

  /// \brief The list of canonical declarations whose redeclaration chains
  /// need to be marked as incomplete once we're done deserializing things.
  SmallVector<Decl *, 16> PendingIncompleteDeclChains;

  /// \brief The Decl IDs for the Sema/Lexical DeclContext of a Decl that has
  /// been loaded but its DeclContext was not set yet.
  struct PendingDeclContextInfo {
    Decl *D;
    serialization::GlobalDeclID SemaDC;
    serialization::GlobalDeclID LexicalDC;
  };

  /// \brief The set of Decls that have been loaded but their DeclContexts are
  /// not set yet.
  ///
  /// The DeclContexts for these Decls will be set once recursive loading has
  /// been completed.
  std::deque<PendingDeclContextInfo> PendingDeclContextInfos;

  /// \brief The set of NamedDecls that have been loaded, but are members of a
  /// context that has been merged into another context where the corresponding
  /// declaration is either missing or has not yet been loaded.
  ///
  /// We will check whether the corresponding declaration is in fact missing
  /// once recursing loading has been completed.
  llvm::SmallVector<NamedDecl *, 16> PendingOdrMergeChecks;

  /// \brief Record definitions in which we found an ODR violation.
  llvm::SmallDenseMap<CXXRecordDecl *, llvm::TinyPtrVector<CXXRecordDecl *>, 2>
      PendingOdrMergeFailures;

  /// \brief DeclContexts in which we have diagnosed an ODR violation.
  llvm::SmallPtrSet<DeclContext*, 2> DiagnosedOdrMergeFailures;

  /// \brief The set of Objective-C categories that have been deserialized
  /// since the last time the declaration chains were linked.
  llvm::SmallPtrSet<ObjCCategoryDecl *, 16> CategoriesDeserialized;

  /// \brief The set of Objective-C class definitions that have already been
  /// loaded, for which we will need to check for categories whenever a new
  /// module is loaded.
  SmallVector<ObjCInterfaceDecl *, 16> ObjCClassesLoaded;

  typedef llvm::DenseMap<Decl *, SmallVector<serialization::DeclID, 2> >
    KeyDeclsMap;
    
  /// \brief A mapping from canonical declarations to the set of global
  /// declaration IDs for key declaration that have been merged with that
  /// canonical declaration. A key declaration is a formerly-canonical
  /// declaration whose module did not import any other key declaration for that
  /// entity. These are the IDs that we use as keys when finding redecl chains.
  KeyDeclsMap KeyDecls;
  
  /// \brief A mapping from DeclContexts to the semantic DeclContext that we
  /// are treating as the definition of the entity. This is used, for instance,
  /// when merging implicit instantiations of class templates across modules.
  llvm::DenseMap<DeclContext *, DeclContext *> MergedDeclContexts;

  /// \brief A mapping from canonical declarations of enums to their canonical
  /// definitions. Only populated when using modules in C++.
  llvm::DenseMap<EnumDecl *, EnumDecl *> EnumDefinitions;

  /// \brief When reading a Stmt tree, Stmt operands are placed in this stack.
  SmallVector<Stmt *, 16> StmtStack;

  /// \brief What kind of records we are reading.
  enum ReadingKind {
    Read_None, Read_Decl, Read_Type, Read_Stmt
  };

  /// \brief What kind of records we are reading.
  ReadingKind ReadingKind;

  /// \brief RAII object to change the reading kind.
  class ReadingKindTracker {
    ASTReader &Reader;
    enum ReadingKind PrevKind;

    ReadingKindTracker(const ReadingKindTracker &) = delete;
    void operator=(const ReadingKindTracker &) = delete;

  public:
    ReadingKindTracker(enum ReadingKind newKind, ASTReader &reader)
      : Reader(reader), PrevKind(Reader.ReadingKind) {
      Reader.ReadingKind = newKind;
    }

    ~ReadingKindTracker() { Reader.ReadingKind = PrevKind; }
  };

  /// \brief RAII object to mark the start of processing updates.
  class ProcessingUpdatesRAIIObj {
    ASTReader &Reader;
    bool PrevState;

    ProcessingUpdatesRAIIObj(const ProcessingUpdatesRAIIObj &) = delete;
    void operator=(const ProcessingUpdatesRAIIObj &) = delete;

  public:
    ProcessingUpdatesRAIIObj(ASTReader &reader)
      : Reader(reader), PrevState(Reader.ProcessingUpdateRecords) {
      Reader.ProcessingUpdateRecords = true;
    }

    ~ProcessingUpdatesRAIIObj() { Reader.ProcessingUpdateRecords = PrevState; }
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

  struct InputFileInfo {
    std::string Filename;
    off_t StoredSize;
    time_t StoredTime;
    bool Overridden;
    bool Transient;
  };

  /// \brief Reads the stored information about an input file.
  InputFileInfo readInputFileInfo(ModuleFile &F, unsigned ID);

  /// \brief Retrieve the file entry and 'overridden' bit for an input
  /// file in the given module file.
  serialization::InputFile getInputFile(ModuleFile &F, unsigned ID,
                                        bool Complain = true);

public:
  void ResolveImportedPath(ModuleFile &M, std::string &Filename);
  static void ResolveImportedPath(std::string &Filename, StringRef Prefix);

  /// \brief Returns the first key declaration for the given declaration. This
  /// is one that is formerly-canonical (or still canonical) and whose module
  /// did not import any other key declaration of the entity.
  Decl *getKeyDeclaration(Decl *D) {
    D = D->getCanonicalDecl();
    if (D->isFromASTFile())
      return D;

    auto I = KeyDecls.find(D);
    if (I == KeyDecls.end() || I->second.empty())
      return D;
    return GetExistingDecl(I->second[0]);
  }
  const Decl *getKeyDeclaration(const Decl *D) {
    return getKeyDeclaration(const_cast<Decl*>(D));
  }

  /// \brief Run a callback on each imported key declaration of \p D.
  template <typename Fn>
  void forEachImportedKeyDecl(const Decl *D, Fn Visit) {
    D = D->getCanonicalDecl();
    if (D->isFromASTFile())
      Visit(D);

    auto It = KeyDecls.find(const_cast<Decl*>(D));
    if (It != KeyDecls.end())
      for (auto ID : It->second)
        Visit(GetExistingDecl(ID));
  }

  /// \brief Get the loaded lookup tables for \p Primary, if any.
  const serialization::reader::DeclContextLookupTable *
  getLoadedLookupTables(DeclContext *Primary) const;

private:
  struct ImportedModule {
    ModuleFile *Mod;
    ModuleFile *ImportedBy;
    SourceLocation ImportLoc;

    ImportedModule(ModuleFile *Mod,
                   ModuleFile *ImportedBy,
                   SourceLocation ImportLoc)
      : Mod(Mod), ImportedBy(ImportedBy), ImportLoc(ImportLoc) { }
  };

  ASTReadResult ReadASTCore(StringRef FileName, ModuleKind Type,
                            SourceLocation ImportLoc, ModuleFile *ImportedBy,
                            SmallVectorImpl<ImportedModule> &Loaded,
                            off_t ExpectedSize, time_t ExpectedModTime,
                            serialization::ASTFileSignature ExpectedSignature,
                            unsigned ClientLoadCapabilities);
  ASTReadResult ReadControlBlock(ModuleFile &F,
                                 SmallVectorImpl<ImportedModule> &Loaded,
                                 const ModuleFile *ImportedBy,
                                 unsigned ClientLoadCapabilities);
  static ASTReadResult ReadOptionsBlock(
      llvm::BitstreamCursor &Stream, unsigned ClientLoadCapabilities,
      bool AllowCompatibleConfigurationMismatch, ASTReaderListener &Listener,
      std::string &SuggestedPredefines, bool ValidateDiagnosticOptions);
  ASTReadResult ReadASTBlock(ModuleFile &F, unsigned ClientLoadCapabilities);
  ASTReadResult ReadExtensionBlock(ModuleFile &F);
  bool ParseLineTable(ModuleFile &F, const RecordData &Record);
  bool ReadSourceManagerBlock(ModuleFile &F);
  llvm::BitstreamCursor &SLocCursorForID(int ID);
  SourceLocation getImportLocation(ModuleFile *F);
  ASTReadResult ReadModuleMapFileBlock(RecordData &Record, ModuleFile &F,
                                       const ModuleFile *ImportedBy,
                                       unsigned ClientLoadCapabilities);
  ASTReadResult ReadSubmoduleBlock(ModuleFile &F,
                                   unsigned ClientLoadCapabilities);
  static bool ParseLanguageOptions(const RecordData &Record, bool Complain,
                                   ASTReaderListener &Listener,
                                   bool AllowCompatibleDifferences);
  static bool ParseTargetOptions(const RecordData &Record, bool Complain,
                                 ASTReaderListener &Listener,
                                 bool AllowCompatibleDifferences);
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
  void readExceptionSpec(ModuleFile &ModuleFile,
                         SmallVectorImpl<QualType> &ExceptionStorage,
                         FunctionProtoType::ExceptionSpecInfo &ESI,
                         const RecordData &Record, unsigned &Index);
  RecordLocation TypeCursorForIndex(unsigned Index);
  void LoadedDecl(unsigned Index, Decl *D);
  Decl *ReadDeclRecord(serialization::DeclID ID);
  void markIncompleteDeclChain(Decl *Canon);

  /// \brief Returns the most recent declaration of a declaration (which must be
  /// of a redeclarable kind) that is either local or has already been loaded
  /// merged into its redecl chain.
  Decl *getMostRecentExistingDecl(Decl *D);

  RecordLocation DeclCursorForID(serialization::DeclID ID,
                                 SourceLocation &Location);
  void loadDeclUpdateRecords(serialization::DeclID ID, Decl *D);
  void loadPendingDeclChain(Decl *D, uint64_t LocalOffset);
  void loadObjCCategories(serialization::GlobalDeclID ID, ObjCInterfaceDecl *D,
                          unsigned PreviousGeneration = 0);

  RecordLocation getLocalBitOffset(uint64_t GlobalOffset);
  uint64_t getGlobalBitOffset(ModuleFile &M, uint32_t LocalOffset);

  /// \brief Returns the first preprocessed entity ID that begins or ends after
  /// \arg Loc.
  serialization::PreprocessedEntityID
  findPreprocessedEntity(SourceLocation Loc, bool EndsAfter) const;

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
  llvm::iterator_range<PreprocessingRecord::iterator>
  getModulePreprocessedEntities(ModuleFile &Mod) const;

  class ModuleDeclIterator
      : public llvm::iterator_adaptor_base<
            ModuleDeclIterator, const serialization::LocalDeclID *,
            std::random_access_iterator_tag, const Decl *, ptrdiff_t,
            const Decl *, const Decl *> {
    ASTReader *Reader;
    ModuleFile *Mod;

  public:
    ModuleDeclIterator()
        : iterator_adaptor_base(nullptr), Reader(nullptr), Mod(nullptr) {}

    ModuleDeclIterator(ASTReader *Reader, ModuleFile *Mod,
                       const serialization::LocalDeclID *Pos)
        : iterator_adaptor_base(Pos), Reader(Reader), Mod(Mod) {}

    value_type operator*() const {
      return Reader->GetDecl(Reader->getGlobalDeclID(*Mod, *I));
    }
    value_type operator->() const { return **this; }

    bool operator==(const ModuleDeclIterator &RHS) const {
      assert(Reader == RHS.Reader && Mod == RHS.Mod);
      return I == RHS.I;
    }
  };

  llvm::iterator_range<ModuleDeclIterator>
  getModuleFileLevelDecls(ModuleFile &Mod);

  void PassInterestingDeclsToConsumer();
  void PassInterestingDeclToConsumer(Decl *D);

  void finishPendingActions();
  void diagnoseOdrViolations();

  void pushExternalDeclIntoScope(NamedDecl *D, DeclarationName Name);

  void addPendingDeclContextInfo(Decl *D,
                                 serialization::GlobalDeclID SemaDC,
                                 serialization::GlobalDeclID LexicalDC) {
    assert(D);
    PendingDeclContextInfo Info = { D, SemaDC, LexicalDC };
    PendingDeclContextInfos.push_back(Info);
  }

  /// \brief Produce an error diagnostic and return true.
  ///
  /// This routine should only be used for fatal errors that have to
  /// do with non-routine failures (e.g., corrupted AST file).
  void Error(StringRef Msg);
  void Error(unsigned DiagID, StringRef Arg1 = StringRef(),
             StringRef Arg2 = StringRef());

  ASTReader(const ASTReader &) = delete;
  void operator=(const ASTReader &) = delete;
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
  /// \param PCHContainerRdr the PCHContainerOperations to use for loading and
  /// creating modules.
  ///
  /// \param Extensions the list of module file extensions that can be loaded
  /// from the AST files.
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
  ///
  /// \param AllowConfigurationMismatch If true, the AST reader will not check
  /// for configuration differences between the AST file and the invocation.
  ///
  /// \param ValidateSystemInputs If true, the AST reader will validate
  /// system input files in addition to user input files. This is only
  /// meaningful if \p DisableValidation is false.
  ///
  /// \param UseGlobalIndex If true, the AST reader will try to load and use
  /// the global module index.
  ///
  /// \param ReadTimer If non-null, a timer used to track the time spent
  /// deserializing.
  ASTReader(Preprocessor &PP, ASTContext &Context,
            const PCHContainerReader &PCHContainerRdr,
            ArrayRef<IntrusiveRefCntPtr<ModuleFileExtension>> Extensions,
            StringRef isysroot = "", bool DisableValidation = false,
            bool AllowASTWithCompilerErrors = false,
            bool AllowConfigurationMismatch = false,
            bool ValidateSystemInputs = false, bool UseGlobalIndex = true,
            std::unique_ptr<llvm::Timer> ReadTimer = {});

  ~ASTReader() override;

  SourceManager &getSourceManager() const { return SourceMgr; }
  FileManager &getFileManager() const { return FileMgr; }
  DiagnosticsEngine &getDiags() const { return Diags; }

  /// \brief Flags that indicate what kind of AST loading failures the client
  /// of the AST reader can directly handle.
  ///
  /// When a client states that it can handle a particular kind of failure,
  /// the AST reader will not emit errors when producing that kind of failure.
  enum LoadFailureCapabilities {
    /// \brief The client can't handle any AST loading failures.
    ARR_None = 0,
    /// \brief The client can handle an AST file that cannot load because it
    /// is missing.
    ARR_Missing = 0x1,
    /// \brief The client can handle an AST file that cannot load because it
    /// is out-of-date relative to its input files.
    ARR_OutOfDate = 0x2,
    /// \brief The client can handle an AST file that cannot load because it
    /// was built with a different version of Clang.
    ARR_VersionMismatch = 0x4,
    /// \brief The client can handle an AST file that cannot load because it's
    /// compiled configuration doesn't match that of the context it was
    /// loaded into.
    ARR_ConfigurationMismatch = 0x8
  };

  /// \brief Load the AST file designated by the given file name.
  ///
  /// \param FileName The name of the AST file to load.
  ///
  /// \param Type The kind of AST being loaded, e.g., PCH, module, main file,
  /// or preamble.
  ///
  /// \param ImportLoc the location where the module file will be considered as
  /// imported from. For non-module AST types it should be invalid.
  ///
  /// \param ClientLoadCapabilities The set of client load-failure
  /// capabilities, represented as a bitset of the enumerators of
  /// LoadFailureCapabilities.
  ASTReadResult ReadAST(StringRef FileName, ModuleKind Type,
                        SourceLocation ImportLoc,
                        unsigned ClientLoadCapabilities);

  /// \brief Make the entities in the given module and any of its (non-explicit)
  /// submodules visible to name lookup.
  ///
  /// \param Mod The module whose names should be made visible.
  ///
  /// \param NameVisibility The level of visibility to give the names in the
  /// module.  Visibility can only be increased over time.
  ///
  /// \param ImportLoc The location at which the import occurs.
  void makeModuleVisible(Module *Mod,
                         Module::NameVisibilityKind NameVisibility,
                         SourceLocation ImportLoc);

  /// \brief Make the names within this set of hidden names visible.
  void makeNamesVisible(const HiddenNames &Names, Module *Owner);

  /// \brief Take the AST callbacks listener.
  std::unique_ptr<ASTReaderListener> takeListener() {
    return std::move(Listener);
  }

  /// \brief Set the AST callbacks listener.
  void setListener(std::unique_ptr<ASTReaderListener> Listener) {
    this->Listener = std::move(Listener);
  }

  /// \brief Add an AST callback listener.
  ///
  /// Takes ownership of \p L.
  void addListener(std::unique_ptr<ASTReaderListener> L) {
    if (Listener)
      L = llvm::make_unique<ChainedASTReaderListener>(std::move(L),
                                                      std::move(Listener));
    Listener = std::move(L);
  }

  /// RAII object to temporarily add an AST callback listener.
  class ListenerScope {
    ASTReader &Reader;
    bool Chained;

  public:
    ListenerScope(ASTReader &Reader, std::unique_ptr<ASTReaderListener> L)
        : Reader(Reader), Chained(false) {
      auto Old = Reader.takeListener();
      if (Old) {
        Chained = true;
        L = llvm::make_unique<ChainedASTReaderListener>(std::move(L),
                                                        std::move(Old));
      }
      Reader.setListener(std::move(L));
    }
    ~ListenerScope() {
      auto New = Reader.takeListener();
      if (Chained)
        Reader.setListener(static_cast<ChainedASTReaderListener *>(New.get())
                               ->takeSecond());
    }
  };

  /// \brief Set the AST deserialization listener.
  void setDeserializationListener(ASTDeserializationListener *Listener,
                                  bool TakeOwnership = false);

  /// \brief Determine whether this AST reader has a global index.
  bool hasGlobalIndex() const { return (bool)GlobalIndex; }

  /// \brief Return global module index.
  GlobalModuleIndex *getGlobalIndex() { return GlobalIndex.get(); }

  /// \brief Reset reader for a reload try.
  void resetForReload() { TriedLoadingGlobalIndex = false; }

  /// \brief Attempts to load the global index.
  ///
  /// \returns true if loading the global index has failed for any reason.
  bool loadGlobalIndex();

  /// \brief Determine whether we tried to load the global index, but failed,
  /// e.g., because it is out-of-date or does not exist.
  bool isGlobalIndexUnavailable() const;
  
  /// \brief Initializes the ASTContext
  void InitializeContext();

  /// \brief Update the state of Sema after loading some additional modules.
  void UpdateSema();

  /// \brief Add in-memory (virtual file) buffer.
  void addInMemoryBuffer(StringRef &FileName,
                         std::unique_ptr<llvm::MemoryBuffer> Buffer) {
    ModuleMgr.addInMemoryBuffer(FileName, std::move(Buffer));
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
  static std::string
  getOriginalSourceFile(const std::string &ASTFileName, FileManager &FileMgr,
                        const PCHContainerReader &PCHContainerRdr,
                        DiagnosticsEngine &Diags);

  /// \brief Read the control block for the named AST file.
  ///
  /// \returns true if an error occurred, false otherwise.
  static bool
  readASTFileControlBlock(StringRef Filename, FileManager &FileMgr,
                          const PCHContainerReader &PCHContainerRdr,
                          bool FindModuleFileExtensions,
                          ASTReaderListener &Listener,
                          bool ValidateDiagnosticOptions);

  /// \brief Determine whether the given AST file is acceptable to load into a
  /// translation unit with the given language and target options.
  static bool isAcceptableASTFile(StringRef Filename, FileManager &FileMgr,
                                  const PCHContainerReader &PCHContainerRdr,
                                  const LangOptions &LangOpts,
                                  const TargetOptions &TargetOpts,
                                  const PreprocessorOptions &PPOpts,
                                  std::string ExistingModuleCachePath);

  /// \brief Returns the suggested contents of the predefines buffer,
  /// which contains a (typically-empty) subset of the predefines
  /// build prior to including the precompiled header.
  const std::string &getSuggestedPredefines() { return SuggestedPredefines; }

  /// \brief Read a preallocated preprocessed entity from the external source.
  ///
  /// \returns null if an error occurred that prevented the preprocessed
  /// entity from being loaded.
  PreprocessedEntity *ReadPreprocessedEntity(unsigned Index) override;

  /// \brief Returns a pair of [Begin, End) indices of preallocated
  /// preprocessed entities that \p Range encompasses.
  std::pair<unsigned, unsigned>
      findPreprocessedEntitiesInRange(SourceRange Range) override;

  /// \brief Optionally returns true or false if the preallocated preprocessed
  /// entity with index \p Index came from file \p FID.
  Optional<bool> isPreprocessedEntityInFileID(unsigned Index,
                                              FileID FID) override;

  /// \brief Read the header file information for the given file entry.
  HeaderFileInfo GetHeaderFileInfo(const FileEntry *FE) override;

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

  /// \brief Reads a TemplateArgumentLocInfo appropriate for the
  /// given TemplateArgument kind.
  TemplateArgumentLocInfo
  GetTemplateArgumentLocInfo(ModuleFile &F, TemplateArgument::ArgKind Kind,
                             const RecordData &Record, unsigned &Idx);

  /// \brief Reads a TemplateArgumentLoc.
  TemplateArgumentLoc
  ReadTemplateArgumentLoc(ModuleFile &F,
                          const RecordData &Record, unsigned &Idx);

  const ASTTemplateArgumentListInfo*
  ReadASTTemplateArgumentListInfo(ModuleFile &F,
                                  const RecordData &Record, unsigned &Index);

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
  ModuleFile *getOwningModuleFile(const Decl *D);

  /// \brief Get the best name we know for the module that owns the given
  /// declaration, or an empty string if the declaration is not from a module.
  std::string getOwningModuleNameForDiagnostic(const Decl *D);

  /// \brief Returns the source location for the decl \p ID.
  SourceLocation getSourceLocationForDeclID(serialization::GlobalDeclID ID);

  /// \brief Resolve a declaration ID into a declaration, potentially
  /// building a new declaration.
  Decl *GetDecl(serialization::DeclID ID);
  Decl *GetExternalDecl(uint32_t ID) override;

  /// \brief Resolve a declaration ID into a declaration. Return 0 if it's not
  /// been loaded yet.
  Decl *GetExistingDecl(serialization::DeclID ID);

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

  /// \brief If any redeclarations of \p D have been imported since it was
  /// last checked, this digs out those redeclarations and adds them to the
  /// redeclaration chain for \p D.
  void CompleteRedeclChain(const Decl *D) override;

  CXXBaseSpecifier *GetExternalCXXBaseSpecifiers(uint64_t Offset) override;

  /// \brief Resolve the offset of a statement into a statement.
  ///
  /// This operation will read a new statement from the external
  /// source each time it is called, and is meant to be used via a
  /// LazyOffsetPtr (which is used by Decls for the body of functions, etc).
  Stmt *GetExternalDeclStmt(uint64_t Offset) override;

  /// ReadBlockAbbrevs - Enter a subblock of the specified BlockID with the
  /// specified cursor.  Read the abbreviations that are at the top of the block
  /// and then leave the cursor pointing into the block.
  static bool ReadBlockAbbrevs(llvm::BitstreamCursor &Cursor, unsigned BlockID);

  /// \brief Finds all the visible declarations with a given name.
  /// The current implementation of this method just loads the entire
  /// lookup table as unmaterialized references.
  bool FindExternalVisibleDeclsByName(const DeclContext *DC,
                                      DeclarationName Name) override;

  /// \brief Read all of the declarations lexically stored in a
  /// declaration context.
  ///
  /// \param DC The declaration context whose declarations will be
  /// read.
  ///
  /// \param IsKindWeWant A predicate indicating which declaration kinds
  /// we are interested in.
  ///
  /// \param Decls Vector that will contain the declarations loaded
  /// from the external source. The caller is responsible for merging
  /// these declarations with any declarations already stored in the
  /// declaration context.
  void
  FindExternalLexicalDecls(const DeclContext *DC,
                           llvm::function_ref<bool(Decl::Kind)> IsKindWeWant,
                           SmallVectorImpl<Decl *> &Decls) override;

  /// \brief Get the decls that are contained in a file in the Offset/Length
  /// range. \p Length can be 0 to indicate a point at \p Offset instead of
  /// a range.
  void FindFileRegionDecls(FileID File, unsigned Offset, unsigned Length,
                           SmallVectorImpl<Decl *> &Decls) override;

  /// \brief Notify ASTReader that we started deserialization of
  /// a decl or type so until FinishedDeserializing is called there may be
  /// decls that are initializing. Must be paired with FinishedDeserializing.
  void StartedDeserializing() override;

  /// \brief Notify ASTReader that we finished the deserialization of
  /// a decl or type. Must be paired with StartedDeserializing.
  void FinishedDeserializing() override;

  /// \brief Function that will be invoked when we begin parsing a new
  /// translation unit involving this external AST source.
  ///
  /// This function will provide all of the external definitions to
  /// the ASTConsumer.
  void StartTranslationUnit(ASTConsumer *Consumer) override;

  /// \brief Print some statistics about AST usage.
  void PrintStats() override;

  /// \brief Dump information about the AST reader to standard error.
  void dump();

  /// Return the amount of memory used by memory buffers, breaking down
  /// by heap-backed versus mmap'ed memory.
  void getMemoryBufferSizes(MemoryBufferSizes &sizes) const override;

  /// \brief Initialize the semantic source with the Sema instance
  /// being used to perform semantic analysis on the abstract syntax
  /// tree.
  void InitializeSema(Sema &S) override;

  /// \brief Inform the semantic consumer that Sema is no longer available.
  void ForgetSema() override { SemaObj = nullptr; }

  /// \brief Retrieve the IdentifierInfo for the named identifier.
  ///
  /// This routine builds a new IdentifierInfo for the given identifier. If any
  /// declarations with this name are visible from translation unit scope, their
  /// declarations will be deserialized and introduced into the declaration
  /// chain of the identifier.
  IdentifierInfo *get(StringRef Name) override;

  /// \brief Retrieve an iterator into the set of all identifiers
  /// in all loaded AST files.
  IdentifierIterator *getIdentifiers() override;

  /// \brief Load the contents of the global method pool for a given
  /// selector.
  void ReadMethodPool(Selector Sel) override;

  /// Load the contents of the global method pool for a given
  /// selector if necessary.
  void updateOutOfDateSelector(Selector Sel) override;

  /// \brief Load the set of namespaces that are known to the external source,
  /// which will be used during typo correction.
  void ReadKnownNamespaces(
                         SmallVectorImpl<NamespaceDecl *> &Namespaces) override;

  void ReadUndefinedButUsed(
      llvm::MapVector<NamedDecl *, SourceLocation> &Undefined) override;

  void ReadMismatchingDeleteExpressions(llvm::MapVector<
      FieldDecl *, llvm::SmallVector<std::pair<SourceLocation, bool>, 4>> &
                                            Exprs) override;

  void ReadTentativeDefinitions(
                            SmallVectorImpl<VarDecl *> &TentativeDefs) override;

  void ReadUnusedFileScopedDecls(
                       SmallVectorImpl<const DeclaratorDecl *> &Decls) override;

  void ReadDelegatingConstructors(
                         SmallVectorImpl<CXXConstructorDecl *> &Decls) override;

  void ReadExtVectorDecls(SmallVectorImpl<TypedefNameDecl *> &Decls) override;

  void ReadUnusedLocalTypedefNameCandidates(
      llvm::SmallSetVector<const TypedefNameDecl *, 4> &Decls) override;

  void ReadReferencedSelectors(
          SmallVectorImpl<std::pair<Selector, SourceLocation> > &Sels) override;

  void ReadWeakUndeclaredIdentifiers(
          SmallVectorImpl<std::pair<IdentifierInfo *, WeakInfo> > &WI) override;

  void ReadUsedVTables(SmallVectorImpl<ExternalVTableUse> &VTables) override;

  void ReadPendingInstantiations(
                 SmallVectorImpl<std::pair<ValueDecl *,
                                           SourceLocation> > &Pending) override;

  void ReadLateParsedTemplates(
      llvm::MapVector<const FunctionDecl *, LateParsedTemplate *> &LPTMap)
      override;

  /// \brief Load a selector from disk, registering its ID if it exists.
  void LoadSelector(Selector Sel);

  void SetIdentifierInfo(unsigned ID, IdentifierInfo *II);
  void SetGloballyVisibleDecls(IdentifierInfo *II,
                               const SmallVectorImpl<uint32_t> &DeclIDs,
                               SmallVectorImpl<Decl *> *Decls = nullptr);

  /// \brief Report a diagnostic.
  DiagnosticBuilder Diag(unsigned DiagID);

  /// \brief Report a diagnostic.
  DiagnosticBuilder Diag(SourceLocation Loc, unsigned DiagID);

  IdentifierInfo *DecodeIdentifierInfo(serialization::IdentifierID ID);

  IdentifierInfo *GetIdentifierInfo(ModuleFile &M, const RecordData &Record,
                                    unsigned &Idx) {
    return DecodeIdentifierInfo(getGlobalIdentifierID(M, Record[Idx++]));
  }

  IdentifierInfo *GetIdentifier(serialization::IdentifierID ID) override {
    // Note that we are loading an identifier.
    Deserializing AnIdentifier(this);

    return DecodeIdentifierInfo(ID);
  }

  IdentifierInfo *getLocalIdentifier(ModuleFile &M, unsigned LocalID);

  serialization::IdentifierID getGlobalIdentifierID(ModuleFile &M,
                                                    unsigned LocalID);

  void resolvePendingMacro(IdentifierInfo *II, const PendingMacroInfo &PMInfo);

  /// \brief Retrieve the macro with the given ID.
  MacroInfo *getMacro(serialization::MacroID ID);

  /// \brief Retrieve the global macro ID corresponding to the given local
  /// ID within the given module file.
  serialization::MacroID getGlobalMacroID(ModuleFile &M, unsigned LocalID);

  /// \brief Read the source location entry with index ID.
  bool ReadSLocEntry(int ID) override;

  /// \brief Retrieve the module import location and module name for the
  /// given source manager entry ID.
  std::pair<SourceLocation, StringRef> getModuleImportLoc(int ID) override;

  /// \brief Retrieve the global submodule ID given a module and its local ID
  /// number.
  serialization::SubmoduleID 
  getGlobalSubmoduleID(ModuleFile &M, unsigned LocalID);
  
  /// \brief Retrieve the submodule that corresponds to a global submodule ID.
  ///
  Module *getSubmodule(serialization::SubmoduleID GlobalID);

  /// \brief Retrieve the module that corresponds to the given module ID.
  ///
  /// Note: overrides method in ExternalASTSource
  Module *getModule(unsigned ID) override;

  /// \brief Retrieve the module file with a given local ID within the specified
  /// ModuleFile.
  ModuleFile *getLocalModuleFile(ModuleFile &M, unsigned ID);

  /// \brief Get an ID for the given module file.
  unsigned getModuleFileID(ModuleFile *M);

  /// \brief Return a descriptor for the corresponding module.
  llvm::Optional<ASTSourceDescriptor> getSourceDescriptor(unsigned ID) override;

  /// \brief Retrieve a selector from the given module with its local ID
  /// number.
  Selector getLocalSelector(ModuleFile &M, unsigned LocalID);

  Selector DecodeSelector(serialization::SelectorID Idx);

  Selector GetExternalSelector(serialization::SelectorID ID) override;
  uint32_t GetNumExternalSelectors() override;

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
  TemplateArgument ReadTemplateArgument(ModuleFile &F, const RecordData &Record,
                                        unsigned &Idx,
                                        bool Canonicalize = false);

  /// \brief Read a template parameter list.
  TemplateParameterList *ReadTemplateParameterList(ModuleFile &F,
                                                   const RecordData &Record,
                                                   unsigned &Idx);

  /// \brief Read a template argument array.
  void ReadTemplateArgumentList(SmallVectorImpl<TemplateArgument> &TemplArgs,
                                ModuleFile &F, const RecordData &Record,
                                unsigned &Idx, bool Canonicalize = false);

  /// \brief Read a UnresolvedSet structure.
  void ReadUnresolvedSet(ModuleFile &F, LazyASTUnresolvedSet &Set,
                         const RecordData &Record, unsigned &Idx);

  /// \brief Read a C++ base specifier.
  CXXBaseSpecifier ReadCXXBaseSpecifier(ModuleFile &F,
                                        const RecordData &Record,unsigned &Idx);

  /// \brief Read a CXXCtorInitializer array.
  CXXCtorInitializer **
  ReadCXXCtorInitializers(ModuleFile &F, const RecordData &Record,
                          unsigned &Idx);

  /// \brief Read the contents of a CXXCtorInitializer array.
  CXXCtorInitializer **GetExternalCXXCtorInitializers(uint64_t Offset) override;

  /// \brief Read a source location from raw form and return it in its
  /// originating module file's source location space.
  SourceLocation ReadUntranslatedSourceLocation(uint32_t Raw) const {
    return SourceLocation::getFromRawEncoding((Raw >> 1) | (Raw << 31));
  }

  /// \brief Read a source location from raw form.
  SourceLocation ReadSourceLocation(ModuleFile &ModuleFile, uint32_t Raw) const {
    SourceLocation Loc = ReadUntranslatedSourceLocation(Raw);
    return TranslateSourceLocation(ModuleFile, Loc);
  }

  /// \brief Translate a source location from another module file's source
  /// location space into ours.
  SourceLocation TranslateSourceLocation(ModuleFile &ModuleFile,
                                         SourceLocation Loc) const {
    assert(ModuleFile.SLocRemap.find(Loc.getOffset()) !=
               ModuleFile.SLocRemap.end() &&
           "Cannot find offset to remap.");
    int Remap = ModuleFile.SLocRemap.find(Loc.getOffset())->second;
    return Loc.getLocWithOffset(Remap);
  }

  /// \brief Read a source location.
  SourceLocation ReadSourceLocation(ModuleFile &ModuleFile,
                                    const RecordDataImpl &Record,
                                    unsigned &Idx) {
    return ReadSourceLocation(ModuleFile, Record[Idx++]);
  }

  /// \brief Read a source range.
  SourceRange ReadSourceRange(ModuleFile &F,
                              const RecordData &Record, unsigned &Idx);

  /// \brief Read an integral value
  llvm::APInt ReadAPInt(const RecordData &Record, unsigned &Idx);

  /// \brief Read a signed integral value
  llvm::APSInt ReadAPSInt(const RecordData &Record, unsigned &Idx);

  /// \brief Read a floating-point value
  llvm::APFloat ReadAPFloat(const RecordData &Record,
                            const llvm::fltSemantics &Sem, unsigned &Idx);

  // \brief Read a string
  static std::string ReadString(const RecordData &Record, unsigned &Idx);

  // \brief Read a path
  std::string ReadPath(ModuleFile &F, const RecordData &Record, unsigned &Idx);

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
    assert(!StmtStack.empty() && "Read too many sub-statements!");
    return StmtStack.pop_back_val();
  }

  /// \brief Reads a sub-expression operand during statement reading.
  Expr *ReadSubExpr();

  /// \brief Reads a token out of a record.
  Token ReadToken(ModuleFile &M, const RecordDataImpl &Record, unsigned &Idx);

  /// \brief Reads the macro record located at the given offset.
  MacroInfo *ReadMacroRecord(ModuleFile &F, uint64_t Offset);

  /// \brief Determine the global preprocessed entity ID that corresponds to
  /// the given local ID within the given module.
  serialization::PreprocessedEntityID
  getGlobalPreprocessedEntityID(ModuleFile &M, unsigned LocalID) const;

  /// \brief Add a macro to deserialize its macro directive history.
  ///
  /// \param II The name of the macro.
  /// \param M The module file.
  /// \param MacroDirectivesOffset Offset of the serialized macro directive
  /// history.
  void addPendingMacro(IdentifierInfo *II, ModuleFile *M,
                       uint64_t MacroDirectivesOffset);

  /// \brief Read the set of macros defined by this external macro source.
  void ReadDefinedMacros() override;

  /// \brief Update an out-of-date identifier.
  void updateOutOfDateIdentifier(IdentifierInfo &II) override;

  /// \brief Note that this identifier is up-to-date.
  void markIdentifierUpToDate(IdentifierInfo *II);

  /// \brief Load all external visible decls in the given DeclContext.
  void completeVisibleDeclsMap(const DeclContext *DC) override;

  /// \brief Retrieve the AST context that this AST reader supplements.
  ASTContext &getContext() { return Context; }

  // \brief Contains the IDs for declarations that were requested before we have
  // access to a Sema object.
  SmallVector<uint64_t, 16> PreloadedDeclIDs;

  /// \brief Retrieve the semantic analysis object used to analyze the
  /// translation unit in which the precompiled header is being
  /// imported.
  Sema *getSema() { return SemaObj; }

  /// \brief Get the identifier resolver used for name lookup / updates
  /// in the translation unit scope. We have one of these even if we don't
  /// have a Sema object.
  IdentifierResolver &getIdResolver();

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
  void ReadComments() override;

  bool isProcessingUpdateRecords() { return ProcessingUpdateRecords; }
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
