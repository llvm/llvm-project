//===-- ClangMove.h - Clang move  -----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_MOVE_CLANGMOVE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_MOVE_CLANGMOVE_H

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace clang {
namespace move {

// FIXME: Make it support more types, e.g. function definitions.
// Currently only support moving class definition.
//
// When moving all class declarations in old header, all code in old.h/cc will
// be moved.
class ClangMoveTool : public ast_matchers::MatchFinder::MatchCallback {
public:
  // Information about the declaration being moved.
  struct MovedDecl {
    // FIXME: Replace Decl with SourceRange to get rid of calculating range for
    // the Decl duplicately.
    const clang::NamedDecl *Decl = nullptr;
    clang::SourceManager *SM = nullptr;
    MovedDecl() = default;
    MovedDecl(const clang::NamedDecl *Decl, clang::SourceManager *SM)
        : Decl(Decl), SM(SM) {}
  };

  struct MoveDefinitionSpec {
    // The list of fully qualified names, e.g. Foo, a::Foo, b::Foo.
    SmallVector<std::string, 4> Names;
    // The file path of old header, can be relative path and absolute path.
    std::string OldHeader;
    // The file path of old cc, can be relative path and absolute path.
    std::string OldCC;
    // The file path of new header, can be relative path and absolute path.
    std::string NewHeader;
    // The file path of new cc, can be relative path and absolute path.
    std::string NewCC;
  };

  ClangMoveTool(
      const MoveDefinitionSpec &MoveSpec,
      std::map<std::string, tooling::Replacements> &FileToReplacements,
      llvm::StringRef OriginalRunningDirectory, llvm::StringRef Style);

  void registerMatchers(ast_matchers::MatchFinder *Finder);

  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

  void onEndOfTranslationUnit() override;

  /// Add #includes from old.h/cc files.
  ///
  /// \param IncludeHeader The name of the file being included, as written in
  /// the source code.
  /// \param IsAngled Whether the file name was enclosed in angle brackets.
  /// \param SearchPath The search path which was used to find the IncludeHeader
  /// in the file system. It can be a relative path or an absolute path.
  /// \param FileName The name of file where the IncludeHeader comes from.
  /// \param IncludeFilenameRange The source range for the written file name in #include
  ///  (i.e. "old.h" for #include "old.h") in old.cc.
  /// \param SM The SourceManager.
  void addIncludes(llvm::StringRef IncludeHeader, bool IsAngled,
                   llvm::StringRef SearchPath, llvm::StringRef FileName,
                   clang::CharSourceRange IncludeFilenameRange,
                   const SourceManager &SM);

  std::vector<MovedDecl> &getMovedDecls() { return MovedDecls; }

  std::vector<MovedDecl> &getRemovedDecls() { return RemovedDecls; }

  llvm::SmallPtrSet<const NamedDecl *, 8> &getUnremovedDeclsInOldHeader() {
    return UnremovedDeclsInOldHeader;
  }

private:
  // Make the Path absolute using the OrignalRunningDirectory if the Path is not
  // an absolute path. An empty Path will result in an empty string.
  std::string makeAbsolutePath(StringRef Path);

  void removeClassDefinitionInOldFiles();
  void moveClassDefinitionToNewFiles();
  void moveAll(SourceManager& SM, StringRef OldFile, StringRef NewFile);

  MoveDefinitionSpec Spec;
  // Stores all MatchCallbacks created by this tool.
  std::vector<std::unique_ptr<ast_matchers::MatchFinder::MatchCallback>>
      MatchCallbacks;
  // The Key is file path, value is the replacements being applied to the file.
  std::map<std::string, tooling::Replacements> &FileToReplacements;
  // All declarations (the class decl being moved, forward decls) that need to
  // be moved/copy to the new files, saving in an AST-visited order.
  std::vector<MovedDecl> MovedDecls;
  // The declarations that needs to be removed in old.cc/h.
  std::vector<MovedDecl> RemovedDecls;
  // The #includes in old_header.h.
  std::vector<std::string> HeaderIncludes;
  // The #includes in old_cc.cc.
  std::vector<std::string> CCIncludes;
  // The original working directory where the local clang-move binary runs.
  //
  // clang-move will change its current working directory to the build
  // directory when analyzing the source file. We save the original working
  // directory in order to get the absolute file path for the fields in Spec.
  std::string OriginalRunningDirectory;
  // The name of a predefined code style.
  std::string FallbackStyle;
  // The unmoved named declarations in old header.
  llvm::SmallPtrSet<const NamedDecl*, 8> UnremovedDeclsInOldHeader;
  /// The source range for the written file name in #include (i.e. "old.h" for
  /// #include "old.h") in old.cc,  including the enclosing quotes or angle
  /// brackets.
  clang::CharSourceRange OldHeaderIncludeRange;
};

class ClangMoveAction : public clang::ASTFrontendAction {
public:
  ClangMoveAction(
      const ClangMoveTool::MoveDefinitionSpec &spec,
      std::map<std::string, tooling::Replacements> &FileToReplacements,
      llvm::StringRef OriginalRunningDirectory, llvm::StringRef FallbackStyle)
      : MoveTool(spec, FileToReplacements, OriginalRunningDirectory,
                 FallbackStyle) {
    MoveTool.registerMatchers(&MatchFinder);
  }

  ~ClangMoveAction() override = default;

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &Compiler,
                    llvm::StringRef InFile) override;

private:
  ast_matchers::MatchFinder MatchFinder;
  ClangMoveTool MoveTool;
};

class ClangMoveActionFactory : public tooling::FrontendActionFactory {
public:
  ClangMoveActionFactory(
      const ClangMoveTool::MoveDefinitionSpec &Spec,
      std::map<std::string, tooling::Replacements> &FileToReplacements,
      llvm::StringRef OriginalRunningDirectory, llvm::StringRef FallbackStyle)
      : Spec(Spec), FileToReplacements(FileToReplacements),
        OriginalRunningDirectory(OriginalRunningDirectory),
        FallbackStyle(FallbackStyle) {}

  clang::FrontendAction *create() override {
    return new ClangMoveAction(Spec, FileToReplacements,
                               OriginalRunningDirectory, FallbackStyle);
  }

private:
  const ClangMoveTool::MoveDefinitionSpec &Spec;
  std::map<std::string, tooling::Replacements> &FileToReplacements;
  std::string OriginalRunningDirectory;
  std::string FallbackStyle;
};

} // namespace move
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_MOVE_CLANGMOVE_H
