//===--- ASTUnit.h - ASTUnit utility ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// ASTUnit utility class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_ASTUNIT_H
#define LLVM_CLANG_FRONTEND_ASTUNIT_H

#include "clang-c/Index.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Lex/ModuleLoader.h"
#include "clang/Lex/PreprocessingRecord.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "clang/Serialization/ASTBitCodes.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/MD5.h"
#include <cassert>
#include <memory>
#include <string>
#include <sys/types.h>
#include <utility>
#include <vector>

namespace llvm {
  class MemoryBuffer;
}

namespace clang {
class Sema;
class ASTContext;
class ASTReader;
class CompilerInvocation;
class CompilerInstance;
class Decl;
class DiagnosticsEngine;
class FileEntry;
class FileManager;
class HeaderSearch;
class MemoryBufferCache;
class Preprocessor;
class PCHContainerOperations;
class PCHContainerReader;
class TargetInfo;
class FrontendAction;
class ASTDeserializationListener;

/// \brief Utility class for loading a ASTContext from an AST file.
///
class ASTUnit : public ModuleLoader {
public:
  struct StandaloneFixIt {
    std::pair<unsigned, unsigned> RemoveRange;
    std::pair<unsigned, unsigned> InsertFromRange;
    std::string CodeToInsert;
    bool BeforePreviousInsertions;
  };

  struct StandaloneDiagnostic {
    unsigned ID;
    DiagnosticsEngine::Level Level;
    std::string Message;
    std::string Filename;
    unsigned LocOffset;
    std::vector<std::pair<unsigned, unsigned> > Ranges;
    std::vector<StandaloneFixIt> FixIts;
  };

private:
  std::shared_ptr<LangOptions>            LangOpts;
  IntrusiveRefCntPtr<DiagnosticsEngine>   Diagnostics;
  IntrusiveRefCntPtr<FileManager>         FileMgr;
  IntrusiveRefCntPtr<SourceManager>       SourceMgr;
  IntrusiveRefCntPtr<MemoryBufferCache>   PCMCache;
  std::unique_ptr<HeaderSearch>           HeaderInfo;
  IntrusiveRefCntPtr<TargetInfo>          Target;
  std::shared_ptr<Preprocessor>           PP;
  IntrusiveRefCntPtr<ASTContext>          Ctx;
  std::shared_ptr<TargetOptions>          TargetOpts;
  std::shared_ptr<HeaderSearchOptions>    HSOpts;
  IntrusiveRefCntPtr<ASTReader> Reader;
  bool HadModuleLoaderFatalFailure;

  struct ASTWriterData;
  std::unique_ptr<ASTWriterData> WriterData;

  FileSystemOptions FileSystemOpts;

  /// \brief The AST consumer that received information about the translation
  /// unit as it was parsed or loaded.
  std::unique_ptr<ASTConsumer> Consumer;

  /// \brief The semantic analysis object used to type-check the translation
  /// unit.
  std::unique_ptr<Sema> TheSema;

  /// Optional owned invocation, just used to make the invocation used in
  /// LoadFromCommandLine available.
  std::shared_ptr<CompilerInvocation> Invocation;

  // OnlyLocalDecls - when true, walking this AST should only visit declarations
  // that come from the AST itself, not from included precompiled headers.
  // FIXME: This is temporary; eventually, CIndex will always do this.
  bool                              OnlyLocalDecls;

  /// \brief Whether to capture any diagnostics produced.
  bool CaptureDiagnostics;

  /// \brief Track whether the main file was loaded from an AST or not.
  bool MainFileIsAST;

  /// \brief What kind of translation unit this AST represents.
  TranslationUnitKind TUKind;

  /// \brief Whether we should time each operation.
  bool WantTiming;

  /// \brief Whether the ASTUnit should delete the remapped buffers.
  bool OwnsRemappedFileBuffers;
  
  /// Track the top-level decls which appeared in an ASTUnit which was loaded
  /// from a source file.
  //
  // FIXME: This is just an optimization hack to avoid deserializing large parts
  // of a PCH file when using the Index library on an ASTUnit loaded from
  // source. In the long term we should make the Index library use efficient and
  // more scalable search mechanisms.
  std::vector<Decl*> TopLevelDecls;

  /// \brief Sorted (by file offset) vector of pairs of file offset/Decl.
  typedef SmallVector<std::pair<unsigned, Decl *>, 64> LocDeclsTy;
  typedef llvm::DenseMap<FileID, LocDeclsTy *> FileDeclsTy;

  /// \brief Map from FileID to the file-level declarations that it contains.
  /// The files and decls are only local (and non-preamble) ones.
  FileDeclsTy FileDecls;
  
  /// The name of the original source file used to generate this ASTUnit.
  std::string OriginalSourceFile;

  /// \brief The set of diagnostics produced when creating the preamble.
  SmallVector<StandaloneDiagnostic, 4> PreambleDiagnostics;

  /// \brief The set of diagnostics produced when creating this
  /// translation unit.
  SmallVector<StoredDiagnostic, 4> StoredDiagnostics;

  /// \brief The set of diagnostics produced when failing to parse, e.g. due
  /// to failure to load the PCH.
  SmallVector<StoredDiagnostic, 4> FailedParseDiagnostics;

  /// \brief The number of stored diagnostics that come from the driver
  /// itself.
  ///
  /// Diagnostics that come from the driver are retained from one parse to
  /// the next.
  unsigned NumStoredDiagnosticsFromDriver;
  
  /// \brief Counter that determines when we want to try building a
  /// precompiled preamble.
  ///
  /// If zero, we will never build a precompiled preamble. Otherwise,
  /// it's treated as a counter that decrements each time we reparse
  /// without the benefit of a precompiled preamble. When it hits 1,
  /// we'll attempt to rebuild the precompiled header. This way, if
  /// building the precompiled preamble fails, we won't try again for
  /// some number of calls.
  unsigned PreambleRebuildCounter;

public:
  class PreambleData {
    const FileEntry *File;
    std::vector<char> Buffer;
    mutable unsigned NumLines;
    
  public:
    PreambleData() : File(nullptr), NumLines(0) { }
    
    void assign(const FileEntry *F, const char *begin, const char *end) {
      File = F;
      Buffer.assign(begin, end);
      NumLines = 0;
    }

    void clear() { Buffer.clear(); File = nullptr; NumLines = 0; }

    size_t size() const { return Buffer.size(); }
    bool empty() const { return Buffer.empty(); }

    const char *getBufferStart() const { return &Buffer[0]; }

    unsigned getNumLines() const {
      if (NumLines)
        return NumLines;
      countLines();
      return NumLines;
    }

    SourceRange getSourceRange(const SourceManager &SM) const {
      SourceLocation FileLoc = SM.getLocForStartOfFile(SM.getPreambleFileID());
      return SourceRange(FileLoc, FileLoc.getLocWithOffset(size()-1));
    }

  private:
    void countLines() const;
  };

  const PreambleData &getPreambleData() const {
    return Preamble;
  }

  /// Data used to determine if a file used in the preamble has been changed.
  struct PreambleFileHash {
    /// All files have size set.
    off_t Size;

    /// Modification time is set for files that are on disk.  For memory
    /// buffers it is zero.
    time_t ModTime;

    /// Memory buffers have MD5 instead of modification time.  We don't
    /// compute MD5 for on-disk files because we hope that modification time is
    /// enough to tell if the file was changed.
    llvm::MD5::MD5Result MD5;

    static PreambleFileHash createForFile(off_t Size, time_t ModTime);
    static PreambleFileHash
    createForMemoryBuffer(const llvm::MemoryBuffer *Buffer);

    friend bool operator==(const PreambleFileHash &LHS,
                           const PreambleFileHash &RHS);

    friend bool operator!=(const PreambleFileHash &LHS,
                           const PreambleFileHash &RHS) {
      return !(LHS == RHS);
    }
  };

private:
  /// \brief The contents of the preamble that has been precompiled to
  /// \c PreambleFile.
  PreambleData Preamble;

  /// \brief Whether the preamble ends at the start of a new line.
  /// 
  /// Used to inform the lexer as to whether it's starting at the beginning of
  /// a line after skipping the preamble.
  bool PreambleEndsAtStartOfLine;

  /// \brief Keeps track of the files that were used when computing the 
  /// preamble, with both their buffer size and their modification time.
  ///
  /// If any of the files have changed from one compile to the next,
  /// the preamble must be thrown away.
  llvm::StringMap<PreambleFileHash> FilesInPreamble;

  /// \brief When non-NULL, this is the buffer used to store the contents of
  /// the main file when it has been padded for use with the precompiled
  /// preamble.
  std::unique_ptr<llvm::MemoryBuffer> SavedMainFileBuffer;

  /// \brief When non-NULL, this is the buffer used to store the
  /// contents of the preamble when it has been padded to build the
  /// precompiled preamble.
  std::unique_ptr<llvm::MemoryBuffer> PreambleBuffer;

  /// \brief The number of warnings that occurred while parsing the preamble.
  ///
  /// This value will be used to restore the state of the \c DiagnosticsEngine
  /// object when re-using the precompiled preamble. Note that only the
  /// number of warnings matters, since we will not save the preamble
  /// when any errors are present.
  unsigned NumWarningsInPreamble;

  /// \brief A list of the serialization ID numbers for each of the top-level
  /// declarations parsed within the precompiled preamble.
  std::vector<serialization::DeclID> TopLevelDeclsInPreamble;
  
  /// \brief Whether we should be caching code-completion results.
  bool ShouldCacheCodeCompletionResults : 1;

  /// \brief Whether to include brief documentation within the set of code
  /// completions cached.
  bool IncludeBriefCommentsInCodeCompletion : 1;

  /// \brief True if non-system source files should be treated as volatile
  /// (likely to change while trying to use them).
  bool UserFilesAreVolatile : 1;
 
  /// \brief The language options used when we load an AST file.
  LangOptions ASTFileLangOpts;

  static void ConfigureDiags(IntrusiveRefCntPtr<DiagnosticsEngine> Diags,
                             ASTUnit &AST, bool CaptureDiagnostics);

  void TranslateStoredDiagnostics(FileManager &FileMgr,
                                  SourceManager &SrcMan,
                      const SmallVectorImpl<StandaloneDiagnostic> &Diags,
                            SmallVectorImpl<StoredDiagnostic> &Out);

  void clearFileLevelDecls();

public:
  /// \brief A cached code-completion result, which may be introduced in one of
  /// many different contexts.
  struct CachedCodeCompletionResult {
    /// \brief The code-completion string corresponding to this completion
    /// result.
    CodeCompletionString *Completion;
    
    /// \brief A bitmask that indicates which code-completion contexts should
    /// contain this completion result.
    ///
    /// The bits in the bitmask correspond to the values of
    /// CodeCompleteContext::Kind. To map from a completion context kind to a
    /// bit, shift 1 by that number of bits. Many completions can occur in
    /// several different contexts.
    uint64_t ShowInContexts;
    
    /// \brief The priority given to this code-completion result.
    unsigned Priority;
    
    /// \brief The libclang cursor kind corresponding to this code-completion 
    /// result.
    CXCursorKind Kind;
    
    /// \brief The availability of this code-completion result.
    CXAvailabilityKind Availability;
    
    /// \brief The simplified type class for a non-macro completion result.
    SimplifiedTypeClass TypeClass;
    
    /// \brief The type of a non-macro completion result, stored as a unique
    /// integer used by the string map of cached completion types.
    ///
    /// This value will be zero if the type is not known, or a unique value
    /// determined by the formatted type string. Se \c CachedCompletionTypes
    /// for more information.
    unsigned Type;
  };
  
  /// \brief Retrieve the mapping from formatted type names to unique type
  /// identifiers.
  llvm::StringMap<unsigned> &getCachedCompletionTypes() { 
    return CachedCompletionTypes; 
  }
  
  /// \brief Retrieve the allocator used to cache global code completions.
  std::shared_ptr<GlobalCodeCompletionAllocator>
  getCachedCompletionAllocator() {
    return CachedCompletionAllocator;
  }

  CodeCompletionTUInfo &getCodeCompletionTUInfo() {
    if (!CCTUInfo)
      CCTUInfo = llvm::make_unique<CodeCompletionTUInfo>(
          std::make_shared<GlobalCodeCompletionAllocator>());
    return *CCTUInfo;
  }

private:
  /// \brief Allocator used to store cached code completions.
  std::shared_ptr<GlobalCodeCompletionAllocator> CachedCompletionAllocator;

  std::unique_ptr<CodeCompletionTUInfo> CCTUInfo;

  /// \brief The set of cached code-completion results.
  std::vector<CachedCodeCompletionResult> CachedCompletionResults;
  
  /// \brief A mapping from the formatted type name to a unique number for that
  /// type, which is used for type equality comparisons.
  llvm::StringMap<unsigned> CachedCompletionTypes;
  
  /// \brief A string hash of the top-level declaration and macro definition 
  /// names processed the last time that we reparsed the file.
  ///
  /// This hash value is used to determine when we need to refresh the 
  /// global code-completion cache.
  unsigned CompletionCacheTopLevelHashValue;

  /// \brief A string hash of the top-level declaration and macro definition 
  /// names processed the last time that we reparsed the precompiled preamble.
  ///
  /// This hash value is used to determine when we need to refresh the 
  /// global code-completion cache after a rebuild of the precompiled preamble.
  unsigned PreambleTopLevelHashValue;

  /// \brief The current hash value for the top-level declaration and macro
  /// definition names
  unsigned CurrentTopLevelHashValue;
  
  /// \brief Bit used by CIndex to mark when a translation unit may be in an
  /// inconsistent state, and is not safe to free.
  unsigned UnsafeToFree : 1;

  /// \brief Cache any "global" code-completion results, so that we can avoid
  /// recomputing them with each completion.
  void CacheCodeCompletionResults();
  
  /// \brief Clear out and deallocate 
  void ClearCachedCompletionResults();
  
  ASTUnit(const ASTUnit &) = delete;
  void operator=(const ASTUnit &) = delete;
  
  explicit ASTUnit(bool MainFileIsAST);

  void CleanTemporaryFiles();
  bool Parse(std::shared_ptr<PCHContainerOperations> PCHContainerOps,
             std::unique_ptr<llvm::MemoryBuffer> OverrideMainBuffer);

  struct ComputedPreamble {
    llvm::MemoryBuffer *Buffer;
    std::unique_ptr<llvm::MemoryBuffer> Owner;
    unsigned Size;
    bool PreambleEndsAtStartOfLine;
    ComputedPreamble(llvm::MemoryBuffer *Buffer,
                     std::unique_ptr<llvm::MemoryBuffer> Owner, unsigned Size,
                     bool PreambleEndsAtStartOfLine)
        : Buffer(Buffer), Owner(std::move(Owner)), Size(Size),
          PreambleEndsAtStartOfLine(PreambleEndsAtStartOfLine) {}
  };
  ComputedPreamble ComputePreamble(CompilerInvocation &Invocation,
                                   unsigned MaxLines);

  std::unique_ptr<llvm::MemoryBuffer> getMainBufferWithPrecompiledPreamble(
      std::shared_ptr<PCHContainerOperations> PCHContainerOps,
      const CompilerInvocation &PreambleInvocationIn, bool AllowRebuild = true,
      unsigned MaxLines = 0);
  void RealizeTopLevelDeclsFromPreamble();

  /// \brief Transfers ownership of the objects (like SourceManager) from
  /// \param CI to this ASTUnit.
  void transferASTDataFromCompilerInstance(CompilerInstance &CI);

  /// \brief Allows us to assert that ASTUnit is not being used concurrently,
  /// which is not supported.
  ///
  /// Clients should create instances of the ConcurrencyCheck class whenever
  /// using the ASTUnit in a way that isn't intended to be concurrent, which is
  /// just about any usage.
  /// Becomes a noop in release mode; only useful for debug mode checking.
  class ConcurrencyState {
    void *Mutex; // a llvm::sys::MutexImpl in debug;

  public:
    ConcurrencyState();
    ~ConcurrencyState();

    void start();
    void finish();
  };
  ConcurrencyState ConcurrencyCheckValue;

public:
  class ConcurrencyCheck {
    ASTUnit &Self;
    
  public:
    explicit ConcurrencyCheck(ASTUnit &Self)
      : Self(Self) 
    { 
      Self.ConcurrencyCheckValue.start();
    }
    ~ConcurrencyCheck() {
      Self.ConcurrencyCheckValue.finish();
    }
  };
  friend class ConcurrencyCheck;

  ~ASTUnit() override;

  bool isMainFileAST() const { return MainFileIsAST; }

  bool isUnsafeToFree() const { return UnsafeToFree; }
  void setUnsafeToFree(bool Value) { UnsafeToFree = Value; }

  const DiagnosticsEngine &getDiagnostics() const { return *Diagnostics; }
  DiagnosticsEngine &getDiagnostics()             { return *Diagnostics; }
  
  const SourceManager &getSourceManager() const { return *SourceMgr; }
        SourceManager &getSourceManager()       { return *SourceMgr; }

  const Preprocessor &getPreprocessor() const { return *PP; }
        Preprocessor &getPreprocessor()       { return *PP; }
  std::shared_ptr<Preprocessor> getPreprocessorPtr() const { return PP; }

  const ASTContext &getASTContext() const { return *Ctx; }
        ASTContext &getASTContext()       { return *Ctx; }

  void setASTContext(ASTContext *ctx) { Ctx = ctx; }
  void setPreprocessor(std::shared_ptr<Preprocessor> pp);

  bool hasSema() const { return (bool)TheSema; }
  Sema &getSema() const { 
    assert(TheSema && "ASTUnit does not have a Sema object!");
    return *TheSema;
  }

  const LangOptions &getLangOpts() const {
    assert(LangOpts && " ASTUnit does not have language options");
    return *LangOpts;
  }
  
  const FileManager &getFileManager() const { return *FileMgr; }
        FileManager &getFileManager()       { return *FileMgr; }

  const FileSystemOptions &getFileSystemOpts() const { return FileSystemOpts; }

  IntrusiveRefCntPtr<ASTReader> getASTReader() const;

  StringRef getOriginalSourceFileName() {
    return OriginalSourceFile;
  }

  ASTMutationListener *getASTMutationListener();
  ASTDeserializationListener *getDeserializationListener();

  /// \brief Add a temporary file that the ASTUnit depends on.
  ///
  /// This file will be erased when the ASTUnit is destroyed.
  void addTemporaryFile(StringRef TempFile);

  bool getOnlyLocalDecls() const { return OnlyLocalDecls; }

  bool getOwnsRemappedFileBuffers() const { return OwnsRemappedFileBuffers; }
  void setOwnsRemappedFileBuffers(bool val) { OwnsRemappedFileBuffers = val; }

  StringRef getMainFileName() const;

  /// \brief If this ASTUnit came from an AST file, returns the filename for it.
  StringRef getASTFileName() const;

  typedef std::vector<Decl *>::iterator top_level_iterator;

  top_level_iterator top_level_begin() {
    assert(!isMainFileAST() && "Invalid call for AST based ASTUnit!");
    if (!TopLevelDeclsInPreamble.empty())
      RealizeTopLevelDeclsFromPreamble();
    return TopLevelDecls.begin();
  }

  top_level_iterator top_level_end() {
    assert(!isMainFileAST() && "Invalid call for AST based ASTUnit!");
    if (!TopLevelDeclsInPreamble.empty())
      RealizeTopLevelDeclsFromPreamble();
    return TopLevelDecls.end();
  }

  std::size_t top_level_size() const {
    assert(!isMainFileAST() && "Invalid call for AST based ASTUnit!");
    return TopLevelDeclsInPreamble.size() + TopLevelDecls.size();
  }

  bool top_level_empty() const {
    assert(!isMainFileAST() && "Invalid call for AST based ASTUnit!");
    return TopLevelDeclsInPreamble.empty() && TopLevelDecls.empty();
  }

  /// \brief Add a new top-level declaration.
  void addTopLevelDecl(Decl *D) {
    TopLevelDecls.push_back(D);
  }

  /// \brief Add a new local file-level declaration.
  void addFileLevelDecl(Decl *D);

  /// \brief Get the decls that are contained in a file in the Offset/Length
  /// range. \p Length can be 0 to indicate a point at \p Offset instead of
  /// a range. 
  void findFileRegionDecls(FileID File, unsigned Offset, unsigned Length,
                           SmallVectorImpl<Decl *> &Decls);

  /// \brief Add a new top-level declaration, identified by its ID in
  /// the precompiled preamble.
  void addTopLevelDeclFromPreamble(serialization::DeclID D) {
    TopLevelDeclsInPreamble.push_back(D);
  }

  /// \brief Retrieve a reference to the current top-level name hash value.
  ///
  /// Note: This is used internally by the top-level tracking action
  unsigned &getCurrentTopLevelHashValue() { return CurrentTopLevelHashValue; }

  /// \brief Get the source location for the given file:line:col triplet.
  ///
  /// The difference with SourceManager::getLocation is that this method checks
  /// whether the requested location points inside the precompiled preamble
  /// in which case the returned source location will be a "loaded" one.
  SourceLocation getLocation(const FileEntry *File,
                             unsigned Line, unsigned Col) const;

  /// \brief Get the source location for the given file:offset pair.
  SourceLocation getLocation(const FileEntry *File, unsigned Offset) const;

  /// \brief If \p Loc is a loaded location from the preamble, returns
  /// the corresponding local location of the main file, otherwise it returns
  /// \p Loc.
  SourceLocation mapLocationFromPreamble(SourceLocation Loc);

  /// \brief If \p Loc is a local location of the main file but inside the
  /// preamble chunk, returns the corresponding loaded location from the
  /// preamble, otherwise it returns \p Loc.
  SourceLocation mapLocationToPreamble(SourceLocation Loc);

  bool isInPreambleFileID(SourceLocation Loc);
  bool isInMainFileID(SourceLocation Loc);
  SourceLocation getStartOfMainFileID();
  SourceLocation getEndOfPreambleFileID();

  /// \see mapLocationFromPreamble.
  SourceRange mapRangeFromPreamble(SourceRange R) {
    return SourceRange(mapLocationFromPreamble(R.getBegin()),
                       mapLocationFromPreamble(R.getEnd()));
  }

  /// \see mapLocationToPreamble.
  SourceRange mapRangeToPreamble(SourceRange R) {
    return SourceRange(mapLocationToPreamble(R.getBegin()),
                       mapLocationToPreamble(R.getEnd()));
  }
  
  // Retrieve the diagnostics associated with this AST
  typedef StoredDiagnostic *stored_diag_iterator;
  typedef const StoredDiagnostic *stored_diag_const_iterator;
  stored_diag_const_iterator stored_diag_begin() const { 
    return StoredDiagnostics.begin(); 
  }
  stored_diag_iterator stored_diag_begin() { 
    return StoredDiagnostics.begin(); 
  }
  stored_diag_const_iterator stored_diag_end() const { 
    return StoredDiagnostics.end(); 
  }
  stored_diag_iterator stored_diag_end() { 
    return StoredDiagnostics.end(); 
  }
  unsigned stored_diag_size() const { return StoredDiagnostics.size(); }

  stored_diag_iterator stored_diag_afterDriver_begin() {
    if (NumStoredDiagnosticsFromDriver > StoredDiagnostics.size())
      NumStoredDiagnosticsFromDriver = 0;
    return StoredDiagnostics.begin() + NumStoredDiagnosticsFromDriver; 
  }

  typedef std::vector<CachedCodeCompletionResult>::iterator
    cached_completion_iterator;
  
  cached_completion_iterator cached_completion_begin() {
    return CachedCompletionResults.begin();
  }

  cached_completion_iterator cached_completion_end() {
    return CachedCompletionResults.end();
  }

  unsigned cached_completion_size() const { 
    return CachedCompletionResults.size(); 
  }

  /// \brief Returns an iterator range for the local preprocessing entities
  /// of the local Preprocessor, if this is a parsed source file, or the loaded
  /// preprocessing entities of the primary module if this is an AST file.
  llvm::iterator_range<PreprocessingRecord::iterator>
  getLocalPreprocessingEntities() const;

  /// \brief Type for a function iterating over a number of declarations.
  /// \returns true to continue iteration and false to abort.
  typedef bool (*DeclVisitorFn)(void *context, const Decl *D);

  /// \brief Iterate over local declarations (locally parsed if this is a parsed
  /// source file or the loaded declarations of the primary module if this is an
  /// AST file).
  /// \returns true if the iteration was complete or false if it was aborted.
  bool visitLocalTopLevelDecls(void *context, DeclVisitorFn Fn);

  /// \brief Get the PCH file if one was included.
  const FileEntry *getPCHFile();

  /// \brief Returns true if the ASTUnit was constructed from a serialized
  /// module file.
  bool isModuleFile();

  std::unique_ptr<llvm::MemoryBuffer>
  getBufferForFile(StringRef Filename, std::string *ErrorStr = nullptr);

  /// \brief Determine what kind of translation unit this AST represents.
  TranslationUnitKind getTranslationUnitKind() const { return TUKind; }

  /// \brief A mapping from a file name to the memory buffer that stores the
  /// remapped contents of that file.
  typedef std::pair<std::string, llvm::MemoryBuffer *> RemappedFile;

  /// \brief Create a ASTUnit. Gets ownership of the passed CompilerInvocation.
  static std::unique_ptr<ASTUnit>
  create(std::shared_ptr<CompilerInvocation> CI,
         IntrusiveRefCntPtr<DiagnosticsEngine> Diags, bool CaptureDiagnostics,
         bool UserFilesAreVolatile);

  /// \brief Create a ASTUnit from an AST file.
  ///
  /// \param Filename - The AST file to load.
  ///
  /// \param PCHContainerRdr - The PCHContainerOperations to use for loading and
  /// creating modules.
  /// \param Diags - The diagnostics engine to use for reporting errors; its
  /// lifetime is expected to extend past that of the returned ASTUnit.
  ///
  /// \returns - The initialized ASTUnit or null if the AST failed to load.
  static std::unique_ptr<ASTUnit> LoadFromASTFile(
      const std::string &Filename, const PCHContainerReader &PCHContainerRdr,
      IntrusiveRefCntPtr<DiagnosticsEngine> Diags,
      const FileSystemOptions &FileSystemOpts, bool UseDebugInfo = false,
      bool OnlyLocalDecls = false, ArrayRef<RemappedFile> RemappedFiles = None,
      bool CaptureDiagnostics = false, bool AllowPCHWithCompilerErrors = false,
      bool UserFilesAreVolatile = false);

private:
  /// \brief Helper function for \c LoadFromCompilerInvocation() and
  /// \c LoadFromCommandLine(), which loads an AST from a compiler invocation.
  ///
  /// \param PrecompilePreambleAfterNParses After how many parses the preamble
  /// of this translation unit should be precompiled, to improve the performance
  /// of reparsing. Set to zero to disable preambles.
  ///
  /// \returns \c true if a catastrophic failure occurred (which means that the
  /// \c ASTUnit itself is invalid), or \c false otherwise.
  bool LoadFromCompilerInvocation(
      std::shared_ptr<PCHContainerOperations> PCHContainerOps,
      unsigned PrecompilePreambleAfterNParses);

public:
  
  /// \brief Create an ASTUnit from a source file, via a CompilerInvocation
  /// object, by invoking the optionally provided ASTFrontendAction. 
  ///
  /// \param CI - The compiler invocation to use; it must have exactly one input
  /// source file. The ASTUnit takes ownership of the CompilerInvocation object.
  ///
  /// \param PCHContainerOps - The PCHContainerOperations to use for loading and
  /// creating modules.
  ///
  /// \param Diags - The diagnostics engine to use for reporting errors; its
  /// lifetime is expected to extend past that of the returned ASTUnit.
  ///
  /// \param Action - The ASTFrontendAction to invoke. Its ownership is not
  /// transferred.
  ///
  /// \param Unit - optionally an already created ASTUnit. Its ownership is not
  /// transferred.
  ///
  /// \param Persistent - if true the returned ASTUnit will be complete.
  /// false means the caller is only interested in getting info through the
  /// provided \see Action.
  ///
  /// \param ErrAST - If non-null and parsing failed without any AST to return
  /// (e.g. because the PCH could not be loaded), this accepts the ASTUnit
  /// mainly to allow the caller to see the diagnostics.
  /// This will only receive an ASTUnit if a new one was created. If an already
  /// created ASTUnit was passed in \p Unit then the caller can check that.
  ///
  static ASTUnit *LoadFromCompilerInvocationAction(
      std::shared_ptr<CompilerInvocation> CI,
      std::shared_ptr<PCHContainerOperations> PCHContainerOps,
      IntrusiveRefCntPtr<DiagnosticsEngine> Diags,
      FrontendAction *Action = nullptr, ASTUnit *Unit = nullptr,
      bool Persistent = true, StringRef ResourceFilesPath = StringRef(),
      bool OnlyLocalDecls = false, bool CaptureDiagnostics = false,
      unsigned PrecompilePreambleAfterNParses = 0,
      bool CacheCodeCompletionResults = false,
      bool IncludeBriefCommentsInCodeCompletion = false,
      bool UserFilesAreVolatile = false,
      std::unique_ptr<ASTUnit> *ErrAST = nullptr);

  /// LoadFromCompilerInvocation - Create an ASTUnit from a source file, via a
  /// CompilerInvocation object.
  ///
  /// \param CI - The compiler invocation to use; it must have exactly one input
  /// source file. The ASTUnit takes ownership of the CompilerInvocation object.
  ///
  /// \param PCHContainerOps - The PCHContainerOperations to use for loading and
  /// creating modules.
  ///
  /// \param Diags - The diagnostics engine to use for reporting errors; its
  /// lifetime is expected to extend past that of the returned ASTUnit.
  //
  // FIXME: Move OnlyLocalDecls, UseBumpAllocator to setters on the ASTUnit, we
  // shouldn't need to specify them at construction time.
  static std::unique_ptr<ASTUnit> LoadFromCompilerInvocation(
      std::shared_ptr<CompilerInvocation> CI,
      std::shared_ptr<PCHContainerOperations> PCHContainerOps,
      IntrusiveRefCntPtr<DiagnosticsEngine> Diags, FileManager *FileMgr,
      bool OnlyLocalDecls = false, bool CaptureDiagnostics = false,
      unsigned PrecompilePreambleAfterNParses = 0,
      TranslationUnitKind TUKind = TU_Complete,
      bool CacheCodeCompletionResults = false,
      bool IncludeBriefCommentsInCodeCompletion = false,
      bool UserFilesAreVolatile = false);

  /// LoadFromCommandLine - Create an ASTUnit from a vector of command line
  /// arguments, which must specify exactly one source file.
  ///
  /// \param ArgBegin - The beginning of the argument vector.
  ///
  /// \param ArgEnd - The end of the argument vector.
  ///
  /// \param PCHContainerOps - The PCHContainerOperations to use for loading and
  /// creating modules.
  ///
  /// \param Diags - The diagnostics engine to use for reporting errors; its
  /// lifetime is expected to extend past that of the returned ASTUnit.
  ///
  /// \param ResourceFilesPath - The path to the compiler resource files.
  ///
  /// \param ModuleFormat - If provided, uses the specific module format.
  ///
  /// \param ErrAST - If non-null and parsing failed without any AST to return
  /// (e.g. because the PCH could not be loaded), this accepts the ASTUnit
  /// mainly to allow the caller to see the diagnostics.
  ///
  // FIXME: Move OnlyLocalDecls, UseBumpAllocator to setters on the ASTUnit, we
  // shouldn't need to specify them at construction time.
  static ASTUnit *LoadFromCommandLine(
      const char **ArgBegin, const char **ArgEnd,
      std::shared_ptr<PCHContainerOperations> PCHContainerOps,
      IntrusiveRefCntPtr<DiagnosticsEngine> Diags, StringRef ResourceFilesPath,
      bool OnlyLocalDecls = false, bool CaptureDiagnostics = false,
      ArrayRef<RemappedFile> RemappedFiles = None,
      bool RemappedFilesKeepOriginalName = true,
      unsigned PrecompilePreambleAfterNParses = 0,
      TranslationUnitKind TUKind = TU_Complete,
      bool CacheCodeCompletionResults = false,
      bool IncludeBriefCommentsInCodeCompletion = false,
      bool AllowPCHWithCompilerErrors = false, bool SkipFunctionBodies = false,
      bool UserFilesAreVolatile = false, bool ForSerialization = false,
      llvm::Optional<StringRef> ModuleFormat = llvm::None,
      std::unique_ptr<ASTUnit> *ErrAST = nullptr);

  /// \brief Reparse the source files using the same command-line options that
  /// were originally used to produce this translation unit.
  ///
  /// \returns True if a failure occurred that causes the ASTUnit not to
  /// contain any translation-unit information, false otherwise.
  bool Reparse(std::shared_ptr<PCHContainerOperations> PCHContainerOps,
               ArrayRef<RemappedFile> RemappedFiles = None);

  /// \brief Perform code completion at the given file, line, and
  /// column within this translation unit.
  ///
  /// \param File The file in which code completion will occur.
  ///
  /// \param Line The line at which code completion will occur.
  ///
  /// \param Column The column at which code completion will occur.
  ///
  /// \param IncludeMacros Whether to include macros in the code-completion 
  /// results.
  ///
  /// \param IncludeCodePatterns Whether to include code patterns (such as a 
  /// for loop) in the code-completion results.
  ///
  /// \param IncludeBriefComments Whether to include brief documentation within
  /// the set of code completions returned.
  ///
  /// FIXME: The Diag, LangOpts, SourceMgr, FileMgr, StoredDiagnostics, and
  /// OwnedBuffers parameters are all disgusting hacks. They will go away.
  void CodeComplete(StringRef File, unsigned Line, unsigned Column,
                    ArrayRef<RemappedFile> RemappedFiles, bool IncludeMacros,
                    bool IncludeCodePatterns, bool IncludeBriefComments,
                    CodeCompleteConsumer &Consumer,
                    std::shared_ptr<PCHContainerOperations> PCHContainerOps,
                    DiagnosticsEngine &Diag, LangOptions &LangOpts,
                    SourceManager &SourceMgr, FileManager &FileMgr,
                    SmallVectorImpl<StoredDiagnostic> &StoredDiagnostics,
                    SmallVectorImpl<const llvm::MemoryBuffer *> &OwnedBuffers);

  /// \brief Save this translation unit to a file with the given name.
  ///
  /// \returns true if there was a file error or false if the save was
  /// successful.
  bool Save(StringRef File);

  /// \brief Serialize this translation unit with the given output stream.
  ///
  /// \returns True if an error occurred, false otherwise.
  bool serialize(raw_ostream &OS);

  ModuleLoadResult loadModule(SourceLocation ImportLoc, ModuleIdPath Path,
                              Module::NameVisibilityKind Visibility,
                              bool IsInclusionDirective) override {
    // ASTUnit doesn't know how to load modules (not that this matters).
    return ModuleLoadResult();
  }

  void makeModuleVisible(Module *Mod, Module::NameVisibilityKind Visibility,
                         SourceLocation ImportLoc) override {}

  GlobalModuleIndex *loadGlobalModuleIndex(SourceLocation TriggerLoc) override
    { return nullptr; }
  bool lookupMissingImports(StringRef Name, SourceLocation TriggerLoc) override
    { return 0; }
};

} // namespace clang

#endif
