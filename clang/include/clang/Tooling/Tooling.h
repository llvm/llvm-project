//===--- Tooling.h - Framework for standalone Clang tools -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements functions to run clang tools standalone instead
//  of running them as a plugin.
//
//  A ClangTool is initialized with a CompilationDatabase and a set of files
//  to run over. The tool will then run a user-specified FrontendAction over
//  all TUs in which the given files are compiled.
//
//  It is also possible to run a FrontendAction over a snippet of code by
//  calling runToolOnCode, which is useful for unit testing.
//
//  Applications that need more fine grained control over how to run
//  multiple FrontendActions over code can use ToolInvocation.
//
//  Example tools:
//  - running clang -fsyntax-only over source code from an editor to get
//    fast syntax checks
//  - running match/replace tools over C++ code
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_TOOLING_H
#define LLVM_CLANG_TOOLING_TOOLING_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LLVM.h"
#include "clang/Driver/Util.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/CompilationDatabase.h"
#include <string>
#include <vector>

namespace clang {

namespace driver {
class Compilation;
} // end namespace driver

class CompilerInvocation;
class SourceManager;
class FrontendAction;

namespace tooling {

/// \brief Interface to generate clang::FrontendActions.
class FrontendActionFactory {
public:
  virtual ~FrontendActionFactory();

  /// \brief Returns a new clang::FrontendAction.
  ///
  /// The caller takes ownership of the returned action.
  virtual clang::FrontendAction *create() = 0;
};

/// \brief Returns a new FrontendActionFactory for a given type.
///
/// T must extend clang::FrontendAction.
///
/// Example:
/// FrontendActionFactory *Factory =
///   newFrontendActionFactory<clang::SyntaxOnlyAction>();
template <typename T>
FrontendActionFactory *newFrontendActionFactory();

/// \brief Called at the end of each source file when used with
/// \c newFrontendActionFactory.
class EndOfSourceFileCallback {
public:
  virtual ~EndOfSourceFileCallback() {}
  virtual void run() = 0;
};

/// \brief Returns a new FrontendActionFactory for any type that provides an
/// implementation of newASTConsumer().
///
/// FactoryT must implement: ASTConsumer *newASTConsumer().
///
/// Example:
/// struct ProvidesASTConsumers {
///   clang::ASTConsumer *newASTConsumer();
/// } Factory;
/// FrontendActionFactory *FactoryAdapter =
///   newFrontendActionFactory(&Factory);
template <typename FactoryT>
inline FrontendActionFactory *newFrontendActionFactory(
    FactoryT *ConsumerFactory, EndOfSourceFileCallback *EndCallback = NULL);

/// \brief Runs (and deletes) the tool on 'Code' with the -fsyntax-only flag.
///
/// \param ToolAction The action to run over the code.
/// \param Code C++ code.
/// \param FileName The file name which 'Code' will be mapped as.
///
/// \return - True if 'ToolAction' was successfully executed.
bool runToolOnCode(clang::FrontendAction *ToolAction, const Twine &Code,
                   const Twine &FileName = "input.cc");

/// \brief Runs (and deletes) the tool on 'Code' with the -fsyntax-only flag and
///        with additional other flags.
///
/// \param ToolAction The action to run over the code.
/// \param Code C++ code.
/// \param Args Additional flags to pass on.
/// \param FileName The file name which 'Code' will be mapped as.
///
/// \return - True if 'ToolAction' was successfully executed.
bool runToolOnCodeWithArgs(clang::FrontendAction *ToolAction, const Twine &Code,
                           const std::vector<std::string> &Args,
                           const Twine &FileName = "input.cc");

/// \brief Utility to run a FrontendAction in a single clang invocation.
class ToolInvocation {
 public:
  /// \brief Create a tool invocation.
  ///
  /// \param CommandLine The command line arguments to clang. Note that clang
  /// uses its binary name (CommandLine[0]) to locate its builtin headers.
  /// Callers have to ensure that they are installed in a compatible location
  /// (see clang driver implementation) or mapped in via mapVirtualFile.
  /// \param ToolAction The action to be executed. Class takes ownership.
  /// \param Files The FileManager used for the execution. Class does not take
  /// ownership.
  ToolInvocation(ArrayRef<std::string> CommandLine, FrontendAction *ToolAction,
                 FileManager *Files);

  /// \brief Map a virtual file to be used while running the tool.
  ///
  /// \param FilePath The path at which the content will be mapped.
  /// \param Content A null terminated buffer of the file's content.
  void mapVirtualFile(StringRef FilePath, StringRef Content);

  /// \brief Run the clang invocation.
  ///
  /// \returns True if there were no errors during execution.
  bool run();

 private:
  void addFileMappingsTo(SourceManager &SourceManager);

  bool runInvocation(const char *BinaryName,
                     clang::driver::Compilation *Compilation,
                     clang::CompilerInvocation *Invocation,
                     const clang::driver::ArgStringList &CC1Args);

  std::vector<std::string> CommandLine;
  llvm::OwningPtr<FrontendAction> ToolAction;
  FileManager *Files;
  // Maps <file name> -> <file content>.
  llvm::StringMap<StringRef> MappedFileContents;
};

/// \brief Utility to run a FrontendAction over a set of files.
///
/// This class is written to be usable for command line utilities.
/// By default the class uses ClangSyntaxOnlyAdjuster to modify
/// command line arguments before the arguments are used to run
/// a frontend action. One could install another command line
/// arguments adjuster by call setArgumentsAdjuster() method.
class ClangTool {
 public:
  /// \brief Constructs a clang tool to run over a list of files.
  ///
  /// \param Compilations The CompilationDatabase which contains the compile
  ///        command lines for the given source paths.
  /// \param SourcePaths The source files to run over. If a source files is
  ///        not found in Compilations, it is skipped.
  ClangTool(const CompilationDatabase &Compilations,
            ArrayRef<std::string> SourcePaths);

  /// \brief Map a virtual file to be used while running the tool.
  ///
  /// \param FilePath The path at which the content will be mapped.
  /// \param Content A null terminated buffer of the file's content.
  void mapVirtualFile(StringRef FilePath, StringRef Content);

  /// \brief Install command line arguments adjuster.
  ///
  /// \param Adjuster Command line arguments adjuster.
  void setArgumentsAdjuster(ArgumentsAdjuster *Adjuster);

  /// Runs a frontend action over all files specified in the command line.
  ///
  /// \param ActionFactory Factory generating the frontend actions. The function
  /// takes ownership of this parameter. A new action is generated for every
  /// processed translation unit.
  int run(FrontendActionFactory *ActionFactory);

  /// \brief Returns the file manager used in the tool.
  ///
  /// The file manager is shared between all translation units.
  FileManager &getFiles() { return Files; }

 private:
  // We store compile commands as pair (file name, compile command).
  std::vector< std::pair<std::string, CompileCommand> > CompileCommands;

  FileManager Files;
  // Contains a list of pairs (<file name>, <file content>).
  std::vector< std::pair<StringRef, StringRef> > MappedFileContents;

  llvm::OwningPtr<ArgumentsAdjuster> ArgsAdjuster;
};

template <typename T>
FrontendActionFactory *newFrontendActionFactory() {
  class SimpleFrontendActionFactory : public FrontendActionFactory {
  public:
    virtual clang::FrontendAction *create() { return new T; }
  };

  return new SimpleFrontendActionFactory;
}

template <typename FactoryT>
inline FrontendActionFactory *newFrontendActionFactory(
    FactoryT *ConsumerFactory, EndOfSourceFileCallback *EndCallback) {
  class FrontendActionFactoryAdapter : public FrontendActionFactory {
  public:
    explicit FrontendActionFactoryAdapter(FactoryT *ConsumerFactory,
                                          EndOfSourceFileCallback *EndCallback)
      : ConsumerFactory(ConsumerFactory), EndCallback(EndCallback) {}

    virtual clang::FrontendAction *create() {
      return new ConsumerFactoryAdaptor(ConsumerFactory, EndCallback);
    }

  private:
    class ConsumerFactoryAdaptor : public clang::ASTFrontendAction {
    public:
      ConsumerFactoryAdaptor(FactoryT *ConsumerFactory,
                             EndOfSourceFileCallback *EndCallback)
        : ConsumerFactory(ConsumerFactory), EndCallback(EndCallback) {}

      clang::ASTConsumer *CreateASTConsumer(clang::CompilerInstance &,
                                            llvm::StringRef) {
        return ConsumerFactory->newASTConsumer();
      }

    protected:
      virtual void EndSourceFileAction() {
        if (EndCallback != NULL)
          EndCallback->run();
        clang::ASTFrontendAction::EndSourceFileAction();
      }

    private:
      FactoryT *ConsumerFactory;
      EndOfSourceFileCallback *EndCallback;
    };
    FactoryT *ConsumerFactory;
    EndOfSourceFileCallback *EndCallback;
  };

  return new FrontendActionFactoryAdapter(ConsumerFactory, EndCallback);
}

/// \brief Returns the absolute path of \c File, by prepending it with
/// the current directory if \c File is not absolute.
///
/// Otherwise returns \c File.
/// If 'File' starts with "./", the returned path will not contain the "./".
/// Otherwise, the returned path will contain the literal path-concatenation of
/// the current directory and \c File.
///
/// The difference to llvm::sys::fs::make_absolute is that we prefer
/// ::getenv("PWD") if available.
/// FIXME: Make this functionality available from llvm::sys::fs and delete
///        this function.
///
/// \param File Either an absolute or relative path.
std::string getAbsolutePath(StringRef File);

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_TOOLING_H
