//===---- ClangLLVMRename.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a refactoring tool to rename variables so that they start with a
// lowercase letter. This tool is intended to be used to rename variables in
// LLVM codebase in which variable names start with an uppercase letter at
// the moment.
//
//   Usage:
//   clang-llmv-rename <file1> <file2> ...
//
// <file1> ... specify the paths of files in the CMake source tree. This
// path is looked up in the compile command database. Here is how to build
// the tool and apply that to all files under "lld" directory.
//
//   $ git clone https://github.com/llvm/llvm-project.git
//   $ mkdir llvm-project/build
//   $ cd llvm-project/build
//   $ cmake -GNinja -DCMAKE_BUILD_TYPE=Release \
//       -DLLVM_ENABLE_PROJECTS='clang-tools-extra;lld' \
//       -DCMAKE_EXPORT_COMPILE_COMMANDS=On \
//       ../llvm
//   $ ninja
//   $ bin/clang-llvm-rename ../lld/**/*.{cpp,h}
//
// For each variable in given files, the tool first check whether the
// variable's definition is in one of given files or not, and rename it if
// and only if it can find a definition of the variable. If the tool cannot
// modify a definition of a variable, it doesn't rename it, in order to keep
// a program compiles.
//
// Note that this tool is not perfect; it doesn't resolve or even detect
// name conflicts caused by renaming. You may need to rename variables
// before using this tool so that your program is free from name conflicts
// due to lowercase/uppercase renaming.
//
//===----------------------------------------------------------------------===//

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/Lexer.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Execution.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Refactoring/AtomicChange.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::tooling;
using namespace llvm;

namespace {
class RenameCallback : public MatchFinder::MatchCallback {
public:
  RenameCallback(
      std::map<std::string, tooling::Replacements> &FileToReplacements,
      ArrayRef<std::string> Paths)
      : FileToReplacements(FileToReplacements) {
    for (StringRef S : Paths)
      InputFiles.insert(canonicalizePath(S));
  }

  // This function is called for each AST pattern matche.
  void run(const MatchFinder::MatchResult &Result) override {
    SourceManager &SM = *Result.SourceManager;

    if (auto *D = Result.Nodes.getNodeAs<VarDecl>("VarDecl")) {
      if (isa<ParmVarDecl>(D))
        return;
      if (isGlobalConst(D))
        return;
      convert(SM, D->getLocation(), D->getName());
      return;
    }

    if (auto *D = Result.Nodes.getNodeAs<ParmVarDecl>("ParmVarDecl")) {
      if (auto *Fn =
              dyn_cast_or_null<FunctionDecl>(D->getParentFunctionOrMethod()))
        if (Fn->isImplicit())
          return;
      convert(SM, D->getLocation(), "");
      return;
    }

    if (auto *D = Result.Nodes.getNodeAs<FieldDecl>("FieldDecl")) {
      convert(SM, D->getLocation(), "");
      return;
    }

    if (auto *D = Result.Nodes.getNodeAs<DeclRefExpr>("DeclRefExpr")) {
      if (!isInGivenFiles(SM, D->getFoundDecl()->getLocation()))
        return;
      if (isa<EnumConstantDecl>(D->getDecl()) ||
          isa<NonTypeTemplateParmDecl>(D->getDecl()))
        return;
      if (auto *Decl = dyn_cast<VarDecl>(D->getFoundDecl()))
        if (isGlobalConst(Decl))
          return;
      if (D->getDecl()->getName().empty())
        return;
      convert(SM, D->getLocation(), "");
      return;
    }

    if (auto *D = Result.Nodes.getNodeAs<MemberExpr>("MemberExpr")) {
      if (!isInGivenFiles(SM, D->getFoundDecl()->getLocation()))
        return;
      if (D->getMemberDecl()->getName().empty())
        return;
      convert(SM, D->getMemberLoc(), D->getMemberDecl()->getName());
      return;
    }

    if (auto *D =
            Result.Nodes.getNodeAs<CXXCtorInitializer>("CXXCtorInitializer")) {
      if (!isInGivenFiles(SM, D->getMemberLocation()))
        return;
      convert(SM, D->getSourceLocation(), D->getMember()->getName());
    }

    if (auto *D = Result.Nodes.getNodeAs<CXXDependentScopeMemberExpr>(
            "CXXDependentScopeMemberExpr")) {
      if (!isInGivenFiles(SM, D->getMemberLoc()))
        return;

      SourceLocation Loc = D->getBeginLoc();
      int Len = Lexer::MeasureTokenLength(Loc, SM, LangOptions());
      std::string Name = std::string(SM.getCharacterData(Loc), Len);
      if (Name == "ELFT" || Name == "RelTy")
        return;

      convert(SM, D->getMemberLoc(),
              D->getMemberNameInfo().getName().getAsString());
    }
  }

  // Setup an AST matcher.
  void registerMatchers(MatchFinder &M) {
    M.addMatcher(varDecl(isExpansionInMainFile()).bind("VarDecl"), this);
    M.addMatcher(parmVarDecl(isExpansionInMainFile()).bind("ParmVarDecl"),
                 this);
    M.addMatcher(fieldDecl(isExpansionInMainFile()).bind("FieldDecl"), this);
    M.addMatcher(declRefExpr(isExpansionInMainFile()).bind("DeclRefExpr"),
                 this);
    M.addMatcher(memberExpr(isExpansionInMainFile()).bind("MemberExpr"), this);
    M.addMatcher(cxxCtorInitializer().bind("CXXCtorInitializer"), this);
    M.addMatcher(cxxDependentScopeMemberExpr(isExpansionInMainFile())
                     .bind("CXXDependentScopeMemberExpr"),
                 this);
  }

private:
  // Returns true if a given source location is in an input file.
  bool isInGivenFiles(SourceManager &SM, SourceLocation Loc) {
    std::string Path = canonicalizePath(SM.getFilename(SM.getSpellingLoc(Loc)));
    return InputFiles.count(Path);
  }

  // Returns true if D is a global variable whose name looks like
  // ALL_UPPERCASE_NAME.
  bool isGlobalConst(const VarDecl *D) {
    return !D->isLocalVarDeclOrParm() &&
           D->getName().find_first_not_of(
               "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_") == StringRef::npos;
  }

  // Renames an identifier so that it starts with a lowercase letter.
  void convert(SourceManager &SM, SourceLocation Loc, StringRef MustMatch) {
    if (!Loc.isValid())
      return;

    SourceLocation SpellingLoc = SM.getSpellingLoc(Loc);
    if (!SM.isInMainFile(SpellingLoc))
      return;

    int Len = Lexer::MeasureTokenLength(SpellingLoc, SM, LangOptions());
    std::string OldName = std::string(SM.getCharacterData(SpellingLoc), Len);

    if (!MustMatch.empty() && OldName != MustMatch)
      return;

    std::string NewName = tolower(OldName);
    tooling::Replacement Repl(SM, SpellingLoc, OldName.size(), NewName);
    if (!Repl.getFilePath().empty())
      consumeError(FileToReplacements[Repl.getFilePath().str()].add(Repl));
  }

  // Returns a new identifier name for a given identifier. Basically, a new
  // name is differnet only at the first character, but for some identifiers
  // such as all capital ones, there are special renaming rules.
  std::string tolower(StringRef S) {
    // This is a list of special renaming rule.
    std::string Ret = StringSwitch<std::string>(S)
                          .Case("MBOrErr", "mbOrErr")
                          .Case("CGProfile", "cgProfile")
                          .Case("IS", "isec")
                          .Case("ISAddr", "isecAddr")
                          .Case("ISLoc", "isecLoc")
                          .Case("ISLimit", "isecLimit")
                          .Case("prevISLimit", "prevIsecLimit")
                          .Case("TSBase", "tsBase")
                          .Case("TSLimit", "tsLimit")
                          .Case("Static", "isStatic")
                          .Case("Pic", "isPic")
                          .Case("Main", "mainPart")
                          .Case("Class", "eqClass")
                          .Default("");
    if (!Ret.empty())
      return Ret;

    // If a new name would be a reserved word, don't rename.
    for (StringRef S2 : {"new", "case", "default"})
      if (S.lower() == S2)
        return std::string(S);

    // Convert all-uppercase names to all-smallcase.
    if (S.find_first_not_of("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") ==
        StringRef::npos)
      return S.lower();

    // They are converted to lowercase names.
    for (StringRef S2 : {"MBRef", "EFlags", "EKind", "EMachine", "OSec", "MBs"})
      if (S == S2)
        return S2.lower();

    // If an identifier starts with one of the folloiwng prefixes, the
    // prefix will be converted to lowercase.
    for (StringRef Prefix : {"LTO", "ARM", "PPC", "RISCV", "LMA", "VMA", "ELF",
                             "ISD", "ABI", "LHS", "RHS", "VFP", "DTP", "OPT_"})
      if (S.starts_with(Prefix))
        return (Prefix.lower() + S.substr(Prefix.size())).str();

    // Otherwise, lowercase the first character.
    Ret = S.str();
    Ret[0] = std::tolower(Ret[0]);
    return Ret;
  }

  // Removes "../" and make a resulting path absolute.
  std::string canonicalizePath(StringRef Path) {
    SmallString<128> S = Path;
    sys::fs::make_absolute(S);
    sys::path::remove_dots(S, /*remove_dot_dot=*/true);
    return std::string(S);
  }

  std::map<std::string, tooling::Replacements> &FileToReplacements;
  StringSet<> InputFiles;
};
} // end anonymous namespace

// Set up the command line options
static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);
static cl::OptionCategory ToolTemplateCategory("tool-template options");

int main(int argc, const char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);

  auto ExpectedParser =
      tooling::CommonOptionsParser::create(argc, argv, ToolTemplateCategory);
  if (!ExpectedParser)
    return 0;
  tooling::CommonOptionsParser &OptParser = ExpectedParser.get();

  const std::vector<std::string> &Files = OptParser.getSourcePathList();
  tooling::RefactoringTool Tool(OptParser.getCompilations(), Files);

  RenameCallback Callback(Tool.getReplacements(), Files);
  MatchFinder Finder;
  Callback.registerMatchers(Finder);

  std::unique_ptr<tooling::FrontendActionFactory> Factory =
      tooling::newFrontendActionFactory(&Finder);
  return Tool.runAndSave(Factory.get());
}
