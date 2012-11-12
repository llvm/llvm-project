//===-- CompilerInstance.h - Clang Compiler Instance ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_COMPILERINSTANCE_H_
#define LLVM_CLANG_FRONTEND_COMPILERINSTANCE_H_

#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/ModuleLoader.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/OwningPtr.h"
#include <cassert>
#include <list>
#include <string>
#include <utility>

namespace llvm {
class raw_fd_ostream;
class Timer;
}

namespace clang {
class ASTContext;
class ASTConsumer;
class ASTReader;
class CodeCompleteConsumer;
class DiagnosticsEngine;
class DiagnosticConsumer;
class ExternalASTSource;
class FileEntry;
class FileManager;
class FrontendAction;
class Module;
class Preprocessor;
class Sema;
class SourceManager;
class TargetInfo;

/// CompilerInstance - Helper class for managing a single instance of the Clang
/// compiler.
///
/// The CompilerInstance serves two purposes:
///  (1) It manages the various objects which are necessary to run the compiler,
///      for example the preprocessor, the target information, and the AST
///      context.
///  (2) It provides utility routines for constructing and manipulating the
///      common Clang objects.
///
/// The compiler instance generally owns the instance of all the objects that it
/// manages. However, clients can still share objects by manually setting the
/// object and retaking ownership prior to destroying the CompilerInstance.
///
/// The compiler instance is intended to simplify clients, but not to lock them
/// in to the compiler instance for everything. When possible, utility functions
/// come in two forms; a short form that reuses the CompilerInstance objects,
/// and a long form that takes explicit instances of any required objects.
class CompilerInstance : public ModuleLoader {
  /// The options used in this compiler instance.
  IntrusiveRefCntPtr<CompilerInvocation> Invocation;

  /// The diagnostics engine instance.
  IntrusiveRefCntPtr<DiagnosticsEngine> Diagnostics;

  /// The target being compiled for.
  IntrusiveRefCntPtr<TargetInfo> Target;

  /// The file manager.
  IntrusiveRefCntPtr<FileManager> FileMgr;

  /// The source manager.
  IntrusiveRefCntPtr<SourceManager> SourceMgr;

  /// The preprocessor.
  IntrusiveRefCntPtr<Preprocessor> PP;

  /// The AST context.
  IntrusiveRefCntPtr<ASTContext> Context;

  /// The AST consumer.
  OwningPtr<ASTConsumer> Consumer;

  /// The code completion consumer.
  OwningPtr<CodeCompleteConsumer> CompletionConsumer;

  /// \brief The semantic analysis object.
  OwningPtr<Sema> TheSema;
  
  /// \brief The frontend timer
  OwningPtr<llvm::Timer> FrontendTimer;

  /// \brief Non-owning reference to the ASTReader, if one exists.
  ASTReader *ModuleManager;

  /// \brief The set of top-level modules that has already been loaded,
  /// along with the module map
  llvm::DenseMap<const IdentifierInfo *, Module *> KnownModules;
  
  /// \brief The location of the module-import keyword for the last module
  /// import. 
  SourceLocation LastModuleImportLoc;
  
  /// \brief The result of the last module import.
  ///
  Module *LastModuleImportResult;
  
  /// \brief Holds information about the output file.
  ///
  /// If TempFilename is not empty we must rename it to Filename at the end.
  /// TempFilename may be empty and Filename non empty if creating the temporary
  /// failed.
  struct OutputFile {
    std::string Filename;
    std::string TempFilename;
    raw_ostream *OS;

    OutputFile(const std::string &filename, const std::string &tempFilename,
               raw_ostream *os)
      : Filename(filename), TempFilename(tempFilename), OS(os) { }
  };

  /// The list of active output files.
  std::list<OutputFile> OutputFiles;

  CompilerInstance(const CompilerInstance &) LLVM_DELETED_FUNCTION;
  void operator=(const CompilerInstance &) LLVM_DELETED_FUNCTION;
public:
  CompilerInstance();
  ~CompilerInstance();

  /// @name High-Level Operations
  /// {

  /// ExecuteAction - Execute the provided action against the compiler's
  /// CompilerInvocation object.
  ///
  /// This function makes the following assumptions:
  ///
  ///  - The invocation options should be initialized. This function does not
  ///    handle the '-help' or '-version' options, clients should handle those
  ///    directly.
  ///
  ///  - The diagnostics engine should have already been created by the client.
  ///
  ///  - No other CompilerInstance state should have been initialized (this is
  ///    an unchecked error).
  ///
  ///  - Clients should have initialized any LLVM target features that may be
  ///    required.
  ///
  ///  - Clients should eventually call llvm_shutdown() upon the completion of
  ///    this routine to ensure that any managed objects are properly destroyed.
  ///
  /// Note that this routine may write output to 'stderr'.
  ///
  /// \param Act - The action to execute.
  /// \return - True on success.
  //
  // FIXME: This function should take the stream to write any debugging /
  // verbose output to as an argument.
  //
  // FIXME: Eliminate the llvm_shutdown requirement, that should either be part
  // of the context or else not CompilerInstance specific.
  bool ExecuteAction(FrontendAction &Act);

  /// }
  /// @name Compiler Invocation and Options
  /// {

  bool hasInvocation() const { return Invocation != 0; }

  CompilerInvocation &getInvocation() {
    assert(Invocation && "Compiler instance has no invocation!");
    return *Invocation;
  }

  /// setInvocation - Replace the current invocation.
  void setInvocation(CompilerInvocation *Value);

  /// }
  /// @name Forwarding Methods
  /// {

  AnalyzerOptionsRef getAnalyzerOpts() {
    return Invocation->getAnalyzerOpts();
  }

  CodeGenOptions &getCodeGenOpts() {
    return Invocation->getCodeGenOpts();
  }
  const CodeGenOptions &getCodeGenOpts() const {
    return Invocation->getCodeGenOpts();
  }

  DependencyOutputOptions &getDependencyOutputOpts() {
    return Invocation->getDependencyOutputOpts();
  }
  const DependencyOutputOptions &getDependencyOutputOpts() const {
    return Invocation->getDependencyOutputOpts();
  }

  DiagnosticOptions &getDiagnosticOpts() {
    return Invocation->getDiagnosticOpts();
  }
  const DiagnosticOptions &getDiagnosticOpts() const {
    return Invocation->getDiagnosticOpts();
  }

  const FileSystemOptions &getFileSystemOpts() const {
    return Invocation->getFileSystemOpts();
  }

  FrontendOptions &getFrontendOpts() {
    return Invocation->getFrontendOpts();
  }
  const FrontendOptions &getFrontendOpts() const {
    return Invocation->getFrontendOpts();
  }

  HeaderSearchOptions &getHeaderSearchOpts() {
    return Invocation->getHeaderSearchOpts();
  }
  const HeaderSearchOptions &getHeaderSearchOpts() const {
    return Invocation->getHeaderSearchOpts();
  }

  LangOptions &getLangOpts() {
    return *Invocation->getLangOpts();
  }
  const LangOptions &getLangOpts() const {
    return *Invocation->getLangOpts();
  }

  PreprocessorOptions &getPreprocessorOpts() {
    return Invocation->getPreprocessorOpts();
  }
  const PreprocessorOptions &getPreprocessorOpts() const {
    return Invocation->getPreprocessorOpts();
  }

  PreprocessorOutputOptions &getPreprocessorOutputOpts() {
    return Invocation->getPreprocessorOutputOpts();
  }
  const PreprocessorOutputOptions &getPreprocessorOutputOpts() const {
    return Invocation->getPreprocessorOutputOpts();
  }

  TargetOptions &getTargetOpts() {
    return Invocation->getTargetOpts();
  }
  const TargetOptions &getTargetOpts() const {
    return Invocation->getTargetOpts();
  }

  /// }
  /// @name Diagnostics Engine
  /// {

  bool hasDiagnostics() const { return Diagnostics != 0; }

  /// Get the current diagnostics engine.
  DiagnosticsEngine &getDiagnostics() const {
    assert(Diagnostics && "Compiler instance has no diagnostics!");
    return *Diagnostics;
  }

  /// setDiagnostics - Replace the current diagnostics engine.
  void setDiagnostics(DiagnosticsEngine *Value);

  DiagnosticConsumer &getDiagnosticClient() const {
    assert(Diagnostics && Diagnostics->getClient() && 
           "Compiler instance has no diagnostic client!");
    return *Diagnostics->getClient();
  }

  /// }
  /// @name Target Info
  /// {

  bool hasTarget() const { return Target != 0; }

  TargetInfo &getTarget() const {
    assert(Target && "Compiler instance has no target!");
    return *Target;
  }

  /// Replace the current diagnostics engine.
  void setTarget(TargetInfo *Value);

  /// }
  /// @name File Manager
  /// {

  bool hasFileManager() const { return FileMgr != 0; }

  /// Return the current file manager to the caller.
  FileManager &getFileManager() const {
    assert(FileMgr && "Compiler instance has no file manager!");
    return *FileMgr;
  }
  
  void resetAndLeakFileManager() {
    FileMgr.resetWithoutRelease();
  }

  /// setFileManager - Replace the current file manager.
  void setFileManager(FileManager *Value);

  /// }
  /// @name Source Manager
  /// {

  bool hasSourceManager() const { return SourceMgr != 0; }

  /// Return the current source manager.
  SourceManager &getSourceManager() const {
    assert(SourceMgr && "Compiler instance has no source manager!");
    return *SourceMgr;
  }
  
  void resetAndLeakSourceManager() {
    SourceMgr.resetWithoutRelease();
  }

  /// setSourceManager - Replace the current source manager.
  void setSourceManager(SourceManager *Value);

  /// }
  /// @name Preprocessor
  /// {

  bool hasPreprocessor() const { return PP != 0; }

  /// Return the current preprocessor.
  Preprocessor &getPreprocessor() const {
    assert(PP && "Compiler instance has no preprocessor!");
    return *PP;
  }

  void resetAndLeakPreprocessor() {
    PP.resetWithoutRelease();
  }

  /// Replace the current preprocessor.
  void setPreprocessor(Preprocessor *Value);

  /// }
  /// @name ASTContext
  /// {

  bool hasASTContext() const { return Context != 0; }

  ASTContext &getASTContext() const {
    assert(Context && "Compiler instance has no AST context!");
    return *Context;
  }
  
  void resetAndLeakASTContext() {
    Context.resetWithoutRelease();
  }

  /// setASTContext - Replace the current AST context.
  void setASTContext(ASTContext *Value);

  /// \brief Replace the current Sema; the compiler instance takes ownership
  /// of S.
  void setSema(Sema *S);
  
  /// }
  /// @name ASTConsumer
  /// {

  bool hasASTConsumer() const { return Consumer != 0; }

  ASTConsumer &getASTConsumer() const {
    assert(Consumer && "Compiler instance has no AST consumer!");
    return *Consumer;
  }

  /// takeASTConsumer - Remove the current AST consumer and give ownership to
  /// the caller.
  ASTConsumer *takeASTConsumer() { return Consumer.take(); }

  /// setASTConsumer - Replace the current AST consumer; the compiler instance
  /// takes ownership of \p Value.
  void setASTConsumer(ASTConsumer *Value);

  /// }
  /// @name Semantic analysis
  /// {
  bool hasSema() const { return TheSema != 0; }
  
  Sema &getSema() const { 
    assert(TheSema && "Compiler instance has no Sema object!");
    return *TheSema;
  }
  
  Sema *takeSema() { return TheSema.take(); }
  
  /// }
  /// @name Module Management
  /// {

  ASTReader *getModuleManager() const { return ModuleManager; }

  /// }
  /// @name Code Completion
  /// {

  bool hasCodeCompletionConsumer() const { return CompletionConsumer != 0; }

  CodeCompleteConsumer &getCodeCompletionConsumer() const {
    assert(CompletionConsumer &&
           "Compiler instance has no code completion consumer!");
    return *CompletionConsumer;
  }

  /// takeCodeCompletionConsumer - Remove the current code completion consumer
  /// and give ownership to the caller.
  CodeCompleteConsumer *takeCodeCompletionConsumer() {
    return CompletionConsumer.take();
  }

  /// setCodeCompletionConsumer - Replace the current code completion consumer;
  /// the compiler instance takes ownership of \p Value.
  void setCodeCompletionConsumer(CodeCompleteConsumer *Value);

  /// }
  /// @name Frontend timer
  /// {

  bool hasFrontendTimer() const { return FrontendTimer != 0; }

  llvm::Timer &getFrontendTimer() const {
    assert(FrontendTimer && "Compiler instance has no frontend timer!");
    return *FrontendTimer;
  }

  /// }
  /// @name Output Files
  /// {

  /// addOutputFile - Add an output file onto the list of tracked output files.
  ///
  /// \param OutFile - The output file info.
  void addOutputFile(const OutputFile &OutFile);

  /// clearOutputFiles - Clear the output file list, destroying the contained
  /// output streams.
  ///
  /// \param EraseFiles - If true, attempt to erase the files from disk.
  void clearOutputFiles(bool EraseFiles);

  /// }
  /// @name Construction Utility Methods
  /// {

  /// Create the diagnostics engine using the invocation's diagnostic options
  /// and replace any existing one with it.
  ///
  /// Note that this routine also replaces the diagnostic client,
  /// allocating one if one is not provided.
  ///
  /// \param Client If non-NULL, a diagnostic client that will be
  /// attached to (and, then, owned by) the DiagnosticsEngine inside this AST
  /// unit.
  ///
  /// \param ShouldOwnClient If Client is non-NULL, specifies whether 
  /// the diagnostic object should take ownership of the client.
  ///
  /// \param ShouldCloneClient If Client is non-NULL, specifies whether that
  /// client should be cloned.
  void createDiagnostics(int Argc, const char* const *Argv,
                         DiagnosticConsumer *Client = 0,
                         bool ShouldOwnClient = true,
                         bool ShouldCloneClient = true);

  /// Create a DiagnosticsEngine object with a the TextDiagnosticPrinter.
  ///
  /// The \p Argc and \p Argv arguments are used only for logging purposes,
  /// when the diagnostic options indicate that the compiler should output
  /// logging information.
  ///
  /// If no diagnostic client is provided, this creates a
  /// DiagnosticConsumer that is owned by the returned diagnostic
  /// object, if using directly the caller is responsible for
  /// releasing the returned DiagnosticsEngine's client eventually.
  ///
  /// \param Opts - The diagnostic options; note that the created text
  /// diagnostic object contains a reference to these options.
  ///
  /// \param Client If non-NULL, a diagnostic client that will be
  /// attached to (and, then, owned by) the returned DiagnosticsEngine
  /// object.
  ///
  /// \param CodeGenOpts If non-NULL, the code gen options in use, which may be
  /// used by some diagnostics printers (for logging purposes only).
  ///
  /// \return The new object on success, or null on failure.
  static IntrusiveRefCntPtr<DiagnosticsEngine>
  createDiagnostics(DiagnosticOptions *Opts, int Argc,
                    const char* const *Argv,
                    DiagnosticConsumer *Client = 0,
                    bool ShouldOwnClient = true,
                    bool ShouldCloneClient = true,
                    const CodeGenOptions *CodeGenOpts = 0);

  /// Create the file manager and replace any existing one with it.
  void createFileManager();

  /// Create the source manager and replace any existing one with it.
  void createSourceManager(FileManager &FileMgr);

  /// Create the preprocessor, using the invocation, file, and source managers,
  /// and replace any existing one with it.
  void createPreprocessor();

  /// Create the AST context.
  void createASTContext();

  /// Create an external AST source to read a PCH file and attach it to the AST
  /// context.
  void createPCHExternalASTSource(StringRef Path,
                                  bool DisablePCHValidation,
                                  bool AllowPCHWithCompilerErrors,
                                  void *DeserializationListener);

  /// Create an external AST source to read a PCH file.
  ///
  /// \return - The new object on success, or null on failure.
  static ExternalASTSource *
  createPCHExternalASTSource(StringRef Path, const std::string &Sysroot,
                             bool DisablePCHValidation,
                             bool AllowPCHWithCompilerErrors,
                             Preprocessor &PP, ASTContext &Context,
                             void *DeserializationListener, bool Preamble);

  /// Create a code completion consumer using the invocation; note that this
  /// will cause the source manager to truncate the input source file at the
  /// completion point.
  void createCodeCompletionConsumer();

  /// Create a code completion consumer to print code completion results, at
  /// \p Filename, \p Line, and \p Column, to the given output stream \p OS.
  static CodeCompleteConsumer *
  createCodeCompletionConsumer(Preprocessor &PP, const std::string &Filename,
                               unsigned Line, unsigned Column,
                               const CodeCompleteOptions &Opts,
                               raw_ostream &OS);

  /// \brief Create the Sema object to be used for parsing.
  void createSema(TranslationUnitKind TUKind,
                  CodeCompleteConsumer *CompletionConsumer);
  
  /// Create the frontend timer and replace any existing one with it.
  void createFrontendTimer();

  /// Create the default output file (from the invocation's options) and add it
  /// to the list of tracked output files.
  ///
  /// The files created by this function always use temporary files to write to
  /// their result (that is, the data is written to a temporary file which will
  /// atomically replace the target output on success).
  ///
  /// \return - Null on error.
  llvm::raw_fd_ostream *
  createDefaultOutputFile(bool Binary = true, StringRef BaseInput = "",
                          StringRef Extension = "");

  /// Create a new output file and add it to the list of tracked output files,
  /// optionally deriving the output path name.
  ///
  /// \return - Null on error.
  llvm::raw_fd_ostream *
  createOutputFile(StringRef OutputPath,
                   bool Binary = true, bool RemoveFileOnSignal = true,
                   StringRef BaseInput = "",
                   StringRef Extension = "",
                   bool UseTemporary = false,
                   bool CreateMissingDirectories = false);

  /// Create a new output file, optionally deriving the output path name.
  ///
  /// If \p OutputPath is empty, then createOutputFile will derive an output
  /// path location as \p BaseInput, with any suffix removed, and \p Extension
  /// appended. If \p OutputPath is not stdout and \p UseTemporary
  /// is true, createOutputFile will create a new temporary file that must be
  /// renamed to \p OutputPath in the end.
  ///
  /// \param OutputPath - If given, the path to the output file.
  /// \param Error [out] - On failure, the error message.
  /// \param BaseInput - If \p OutputPath is empty, the input path name to use
  /// for deriving the output path.
  /// \param Extension - The extension to use for derived output names.
  /// \param Binary - The mode to open the file in.
  /// \param RemoveFileOnSignal - Whether the file should be registered with
  /// llvm::sys::RemoveFileOnSignal. Note that this is not safe for
  /// multithreaded use, as the underlying signal mechanism is not reentrant
  /// \param UseTemporary - Create a new temporary file that must be renamed to
  /// OutputPath in the end.
  /// \param CreateMissingDirectories - When \p UseTemporary is true, create
  /// missing directories in the output path.
  /// \param ResultPathName [out] - If given, the result path name will be
  /// stored here on success.
  /// \param TempPathName [out] - If given, the temporary file path name
  /// will be stored here on success.
  static llvm::raw_fd_ostream *
  createOutputFile(StringRef OutputPath, std::string &Error,
                   bool Binary = true, bool RemoveFileOnSignal = true,
                   StringRef BaseInput = "",
                   StringRef Extension = "",
                   bool UseTemporary = false,
                   bool CreateMissingDirectories = false,
                   std::string *ResultPathName = 0,
                   std::string *TempPathName = 0);

  /// }
  /// @name Initialization Utility Methods
  /// {

  /// InitializeSourceManager - Initialize the source manager to set InputFile
  /// as the main file.
  ///
  /// \return True on success.
  bool InitializeSourceManager(const FrontendInputFile &Input);

  /// InitializeSourceManager - Initialize the source manager to set InputFile
  /// as the main file.
  ///
  /// \return True on success.
  static bool InitializeSourceManager(const FrontendInputFile &Input,
                DiagnosticsEngine &Diags,
                FileManager &FileMgr,
                SourceManager &SourceMgr,
                const FrontendOptions &Opts);

  /// }
  
  virtual Module *loadModule(SourceLocation ImportLoc, ModuleIdPath Path,
                             Module::NameVisibilityKind Visibility,
                             bool IsInclusionDirective);
};

} // end namespace clang

#endif
