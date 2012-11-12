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

#include "clang/Serialization/ASTBitCodes.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "clang/Lex/ModuleLoader.h"
#include "clang/Lex/PreprocessingRecord.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/TargetOptions.h"
#include "clang-c/Index.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Path.h"
#include <map>
#include <string>
#include <vector>
#include <cassert>
#include <utility>
#include <sys/types.h>

namespace llvm {
  class MemoryBuffer;
}

namespace clang {
class ASTContext;
class ASTReader;
class CodeCompleteConsumer;
class CompilerInvocation;
class CompilerInstance;
class Decl;
class DiagnosticsEngine;
class FileEntry;
class FileManager;
class HeaderSearch;
class Preprocessor;
class SourceManager;
class TargetInfo;
class ASTFrontendAction;
class ASTDeserializationListener;

/// \brief Utility class for loading a ASTContext from an AST file.
///
class ASTUnit : public ModuleLoader {
private:
  IntrusiveRefCntPtr<LangOptions>         LangOpts;
  IntrusiveRefCntPtr<DiagnosticsEngine>   Diagnostics;
  IntrusiveRefCntPtr<FileManager>         FileMgr;
  IntrusiveRefCntPtr<SourceManager>       SourceMgr;
  OwningPtr<HeaderSearch>                 HeaderInfo;
  IntrusiveRefCntPtr<TargetInfo>          Target;
  IntrusiveRefCntPtr<Preprocessor>        PP;
  IntrusiveRefCntPtr<ASTContext>          Ctx;
  IntrusiveRefCntPtr<TargetOptions>       TargetOpts;
  IntrusiveRefCntPtr<HeaderSearchOptions> HSOpts;
  ASTReader *Reader;

  struct ASTWriterData;
  OwningPtr<ASTWriterData> WriterData;

  FileSystemOptions FileSystemOpts;

  /// \brief The AST consumer that received information about the translation
  /// unit as it was parsed or loaded.
  OwningPtr<ASTConsumer> Consumer;
  
  /// \brief The semantic analysis object used to type-check the translation
  /// unit.
  OwningPtr<Sema> TheSema;
  
  /// Optional owned invocation, just used to make the invocation used in
  /// LoadFromCommandLine available.
  IntrusiveRefCntPtr<CompilerInvocation> Invocation;
  
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
  SmallVector<StoredDiagnostic, 4> PreambleDiagnostics;

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
    PreambleData() : File(0), NumLines(0) { }
    
    void assign(const FileEntry *F, const char *begin, const char *end) {
      File = F;
      Buffer.assign(begin, end);
      NumLines = 0;
    }

    void clear() { Buffer.clear(); File = 0; NumLines = 0; }

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

private:

  /// \brief The contents of the preamble that has been precompiled to
  /// \c PreambleFile.
  PreambleData Preamble;

  /// \brief Whether the preamble ends at the start of a new line.
  /// 
  /// Used to inform the lexer as to whether it's starting at the beginning of
  /// a line after skipping the preamble.
  bool PreambleEndsAtStartOfLine;
  
  /// \brief The size of the source buffer that we've reserved for the main 
  /// file within the precompiled preamble.
  unsigned PreambleReservedSize;

  /// \brief Keeps track of the files that were used when computing the 
  /// preamble, with both their buffer size and their modification time.
  ///
  /// If any of the files have changed from one compile to the next,
  /// the preamble must be thrown away.
  llvm::StringMap<std::pair<off_t, time_t> > FilesInPreamble;

  /// \brief When non-NULL, this is the buffer used to store the contents of
  /// the main file when it has been padded for use with the precompiled
  /// preamble.
  llvm::MemoryBuffer *SavedMainFileBuffer;

  /// \brief When non-NULL, this is the buffer used to store the
  /// contents of the preamble when it has been padded to build the
  /// precompiled preamble.
  llvm::MemoryBuffer *PreambleBuffer;

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

  static void ConfigureDiags(IntrusiveRefCntPtr<DiagnosticsEngine> &Diags,
                             const char **ArgBegin, const char **ArgEnd,
                             ASTUnit &AST, bool CaptureDiagnostics);

  void TranslateStoredDiagnostics(ASTReader *MMan, StringRef ModName,
                                  SourceManager &SrcMan,
                      const SmallVectorImpl<StoredDiagnostic> &Diags,
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
  IntrusiveRefCntPtr<GlobalCodeCompletionAllocator>
  getCachedCompletionAllocator() {
    return CachedCompletionAllocator;
  }

  CodeCompletionTUInfo &getCodeCompletionTUInfo() {
    if (!CCTUInfo)
      CCTUInfo.reset(new CodeCompletionTUInfo(
                                            new GlobalCodeCompletionAllocator));
    return *CCTUInfo;
  }

private:
  /// \brief Allocator used to store cached code completions.
  IntrusiveRefCntPtr<GlobalCodeCompletionAllocator>
    CachedCompletionAllocator;
  
  OwningPtr<CodeCompletionTUInfo> CCTUInfo;

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
  
  ASTUnit(const ASTUnit &) LLVM_DELETED_FUNCTION;
  void operator=(const ASTUnit &) LLVM_DELETED_FUNCTION;
  
  explicit ASTUnit(bool MainFileIsAST);

  void CleanTemporaryFiles();
  bool Parse(llvm::MemoryBuffer *OverrideMainBuffer);
  
  std::pair<llvm::MemoryBuffer *, std::pair<unsigned, bool> >
  ComputePreamble(CompilerInvocation &Invocation, 
                  unsigned MaxLines, bool &CreatedBuffer);
  
  llvm::MemoryBuffer *getMainBufferWithPrecompiledPreamble(
                               const CompilerInvocation &PreambleInvocationIn,
                                                     bool AllowRebuild = true,
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
#ifndef NDEBUG
    void *Mutex; // a llvm::sys::MutexImpl in debug;
#endif

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
  
  ~ASTUnit();

  bool isMainFileAST() const { return MainFileIsAST; }

  bool isUnsafeToFree() const { return UnsafeToFree; }
  void setUnsafeToFree(bool Value) { UnsafeToFree = Value; }

  const DiagnosticsEngine &getDiagnostics() const { return *Diagnostics; }
  DiagnosticsEngine &getDiagnostics()             { return *Diagnostics; }
  
  const SourceManager &getSourceManager() const { return *SourceMgr; }
        SourceManager &getSourceManager()       { return *SourceMgr; }

  const Preprocessor &getPreprocessor() const { return *PP; }
        Preprocessor &getPreprocessor()       { return *PP; }

  const ASTContext &getASTContext() const { return *Ctx; }
        ASTContext &getASTContext()       { return *Ctx; }

  void setASTContext(ASTContext *ctx) { Ctx = ctx; }
  void setPreprocessor(Preprocessor *pp);

  bool hasSema() const { return TheSema; }
  Sema &getSema() const { 
    assert(TheSema && "ASTUnit does not have a Sema object!");
    return *TheSema; 
  }
  
  const FileManager &getFileManager() const { return *FileMgr; }
        FileManager &getFileManager()       { return *FileMgr; }

  const FileSystemOptions &getFileSystemOpts() const { return FileSystemOpts; }

  StringRef getOriginalSourceFileName() {
    return OriginalSourceFile;
  }

  ASTDeserializationListener *getDeserializationListener();

  /// \brief Add a temporary file that the ASTUnit depends on.
  ///
  /// This file will be erased when the ASTUnit is destroyed.
  void addTemporaryFile(const llvm::sys::Path &TempFile);
                        
  bool getOnlyLocalDecls() const { return OnlyLocalDecls; }

  bool getOwnsRemappedFileBuffers() const { return OwnsRemappedFileBuffers; }
  void setOwnsRemappedFileBuffers(bool val) { OwnsRemappedFileBuffers = val; }

  StringRef getMainFileName() const;

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
  std::pair<PreprocessingRecord::iterator, PreprocessingRecord::iterator>
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

  llvm::MemoryBuffer *getBufferForFile(StringRef Filename,
                                       std::string *ErrorStr = 0);

  /// \brief Determine what kind of translation unit this AST represents.
  TranslationUnitKind getTranslationUnitKind() const { return TUKind; }

  typedef llvm::PointerUnion<const char *, const llvm::MemoryBuffer *>
      FilenameOrMemBuf;
  /// \brief A mapping from a file name to the memory buffer that stores the
  /// remapped contents of that file.
  typedef std::pair<std::string, FilenameOrMemBuf> RemappedFile;

  /// \brief Create a ASTUnit. Gets ownership of the passed CompilerInvocation. 
  static ASTUnit *create(CompilerInvocation *CI,
                         IntrusiveRefCntPtr<DiagnosticsEngine> Diags,
                         bool CaptureDiagnostics,
                         bool UserFilesAreVolatile);

  /// \brief Create a ASTUnit from an AST file.
  ///
  /// \param Filename - The AST file to load.
  ///
  /// \param Diags - The diagnostics engine to use for reporting errors; its
  /// lifetime is expected to extend past that of the returned ASTUnit.
  ///
  /// \returns - The initialized ASTUnit or null if the AST failed to load.
  static ASTUnit *LoadFromASTFile(const std::string &Filename,
                              IntrusiveRefCntPtr<DiagnosticsEngine> Diags,
                                  const FileSystemOptions &FileSystemOpts,
                                  bool OnlyLocalDecls = false,
                                  RemappedFile *RemappedFiles = 0,
                                  unsigned NumRemappedFiles = 0,
                                  bool CaptureDiagnostics = false,
                                  bool AllowPCHWithCompilerErrors = false,
                                  bool UserFilesAreVolatile = false);

private:
  /// \brief Helper function for \c LoadFromCompilerInvocation() and
  /// \c LoadFromCommandLine(), which loads an AST from a compiler invocation.
  ///
  /// \param PrecompilePreamble Whether to precompile the preamble of this
  /// translation unit, to improve the performance of reparsing.
  ///
  /// \returns \c true if a catastrophic failure occurred (which means that the
  /// \c ASTUnit itself is invalid), or \c false otherwise.
  bool LoadFromCompilerInvocation(bool PrecompilePreamble);
  
public:
  
  /// \brief Create an ASTUnit from a source file, via a CompilerInvocation
  /// object, by invoking the optionally provided ASTFrontendAction. 
  ///
  /// \param CI - The compiler invocation to use; it must have exactly one input
  /// source file. The ASTUnit takes ownership of the CompilerInvocation object.
  ///
  /// \param Diags - The diagnostics engine to use for reporting errors; its
  /// lifetime is expected to extend past that of the returned ASTUnit.
  ///
  /// \param Action - The ASTFrontendAction to invoke. Its ownership is not
  /// transfered.
  ///
  /// \param Unit - optionally an already created ASTUnit. Its ownership is not
  /// transfered.
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
  static ASTUnit *LoadFromCompilerInvocationAction(CompilerInvocation *CI,
                              IntrusiveRefCntPtr<DiagnosticsEngine> Diags,
                                             ASTFrontendAction *Action = 0,
                                             ASTUnit *Unit = 0,
                                             bool Persistent = true,
                                      StringRef ResourceFilesPath = StringRef(),
                                             bool OnlyLocalDecls = false,
                                             bool CaptureDiagnostics = false,
                                             bool PrecompilePreamble = false,
                                       bool CacheCodeCompletionResults = false,
                              bool IncludeBriefCommentsInCodeCompletion = false,
                                       bool UserFilesAreVolatile = false,
                                       OwningPtr<ASTUnit> *ErrAST = 0);

  /// LoadFromCompilerInvocation - Create an ASTUnit from a source file, via a
  /// CompilerInvocation object.
  ///
  /// \param CI - The compiler invocation to use; it must have exactly one input
  /// source file. The ASTUnit takes ownership of the CompilerInvocation object.
  ///
  /// \param Diags - The diagnostics engine to use for reporting errors; its
  /// lifetime is expected to extend past that of the returned ASTUnit.
  //
  // FIXME: Move OnlyLocalDecls, UseBumpAllocator to setters on the ASTUnit, we
  // shouldn't need to specify them at construction time.
  static ASTUnit *LoadFromCompilerInvocation(CompilerInvocation *CI,
                              IntrusiveRefCntPtr<DiagnosticsEngine> Diags,
                                             bool OnlyLocalDecls = false,
                                             bool CaptureDiagnostics = false,
                                             bool PrecompilePreamble = false,
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
  /// \param Diags - The diagnostics engine to use for reporting errors; its
  /// lifetime is expected to extend past that of the returned ASTUnit.
  ///
  /// \param ResourceFilesPath - The path to the compiler resource files.
  ///
  /// \param ErrAST - If non-null and parsing failed without any AST to return
  /// (e.g. because the PCH could not be loaded), this accepts the ASTUnit
  /// mainly to allow the caller to see the diagnostics.
  ///
  // FIXME: Move OnlyLocalDecls, UseBumpAllocator to setters on the ASTUnit, we
  // shouldn't need to specify them at construction time.
  static ASTUnit *LoadFromCommandLine(const char **ArgBegin,
                                      const char **ArgEnd,
                              IntrusiveRefCntPtr<DiagnosticsEngine> Diags,
                                      StringRef ResourceFilesPath,
                                      bool OnlyLocalDecls = false,
                                      bool CaptureDiagnostics = false,
                                      RemappedFile *RemappedFiles = 0,
                                      unsigned NumRemappedFiles = 0,
                                      bool RemappedFilesKeepOriginalName = true,
                                      bool PrecompilePreamble = false,
                                      TranslationUnitKind TUKind = TU_Complete,
                                      bool CacheCodeCompletionResults = false,
                            bool IncludeBriefCommentsInCodeCompletion = false,
                                      bool AllowPCHWithCompilerErrors = false,
                                      bool SkipFunctionBodies = false,
                                      bool UserFilesAreVolatile = false,
                                      bool ForSerialization = false,
                                      OwningPtr<ASTUnit> *ErrAST = 0);
  
  /// \brief Reparse the source files using the same command-line options that
  /// were originally used to produce this translation unit.
  ///
  /// \returns True if a failure occurred that causes the ASTUnit not to
  /// contain any translation-unit information, false otherwise.  
  bool Reparse(RemappedFile *RemappedFiles = 0,
               unsigned NumRemappedFiles = 0);

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
                    RemappedFile *RemappedFiles, unsigned NumRemappedFiles,
                    bool IncludeMacros, bool IncludeCodePatterns,
                    bool IncludeBriefComments,
                    CodeCompleteConsumer &Consumer,
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
  
  virtual Module *loadModule(SourceLocation ImportLoc, ModuleIdPath Path,
                             Module::NameVisibilityKind Visibility,
                             bool IsInclusionDirective) {
    // ASTUnit doesn't know how to load modules (not that this matters).
    return 0;
  }
};

} // namespace clang

#endif
