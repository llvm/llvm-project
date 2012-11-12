//===- examples/Tooling/RemoveCStrCalls.cpp - Redundant c_str call removal ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements a tool that prints replacements that remove redundant
//  calls of c_str() on strings.
//
//  Usage:
//  remove-cstr-calls <cmake-output-dir> <file1> <file2> ...
//
//  Where <cmake-output-dir> is a CMake build directory in which a file named
//  compile_commands.json exists (enable -DCMAKE_EXPORT_COMPILE_COMMANDS in
//  CMake to get this output).
//
//  <file1> ... specify the paths of files in the CMake source tree. This path
//  is looked up in the compile command database. If the path of a file is
//  absolute, it needs to point into CMake's source tree. If the path is
//  relative, the current working directory needs to be in the CMake source
//  tree and the file must be in a subdirectory of the current working
//  directory. "./" prefixes in the relative files will be automatically
//  removed, but the rest of a relative path must be a suffix of a path in
//  the compile command line database.
//
//  For example, to use remove-cstr-calls on all files in a subtree of the
//  source tree, use:
//
//    /path/in/subtree $ find . -name '*.cpp'|
//        xargs remove-cstr-calls /path/to/build
//
//===----------------------------------------------------------------------===//

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace llvm;
using clang::tooling::newFrontendActionFactory;
using clang::tooling::Replacement;
using clang::tooling::CompilationDatabase;

// FIXME: Pull out helper methods in here into more fitting places.

// Returns the text that makes up 'node' in the source.
// Returns an empty string if the text cannot be found.
template <typename T>
static std::string getText(const SourceManager &SourceManager, const T &Node) {
  SourceLocation StartSpellingLocatino =
      SourceManager.getSpellingLoc(Node.getLocStart());
  SourceLocation EndSpellingLocation =
      SourceManager.getSpellingLoc(Node.getLocEnd());
  if (!StartSpellingLocatino.isValid() || !EndSpellingLocation.isValid()) {
    return std::string();
  }
  bool Invalid = true;
  const char *Text =
      SourceManager.getCharacterData(StartSpellingLocatino, &Invalid);
  if (Invalid) {
    return std::string();
  }
  std::pair<FileID, unsigned> Start =
      SourceManager.getDecomposedLoc(StartSpellingLocatino);
  std::pair<FileID, unsigned> End =
      SourceManager.getDecomposedLoc(Lexer::getLocForEndOfToken(
          EndSpellingLocation, 0, SourceManager, LangOptions()));
  if (Start.first != End.first) {
    // Start and end are in different files.
    return std::string();
  }
  if (End.second < Start.second) {
    // Shuffling text with macros may cause this.
    return std::string();
  }
  return std::string(Text, End.second - Start.second);
}

// Return true if expr needs to be put in parens when it is an
// argument of a prefix unary operator, e.g. when it is a binary or
// ternary operator syntactically.
static bool needParensAfterUnaryOperator(const Expr &ExprNode) {
  if (dyn_cast<clang::BinaryOperator>(&ExprNode) ||
      dyn_cast<clang::ConditionalOperator>(&ExprNode)) {
    return true;
  }
  if (const CXXOperatorCallExpr *op =
      dyn_cast<CXXOperatorCallExpr>(&ExprNode)) {
    return op->getNumArgs() == 2 &&
        op->getOperator() != OO_PlusPlus &&
        op->getOperator() != OO_MinusMinus &&
        op->getOperator() != OO_Call &&
        op->getOperator() != OO_Subscript;
  }
  return false;
}

// Format a pointer to an expression: prefix with '*' but simplify
// when it already begins with '&'.  Return empty string on failure.
static std::string formatDereference(const SourceManager &SourceManager,
                              const Expr &ExprNode) {
  if (const clang::UnaryOperator *Op =
      dyn_cast<clang::UnaryOperator>(&ExprNode)) {
    if (Op->getOpcode() == UO_AddrOf) {
      // Strip leading '&'.
      return getText(SourceManager, *Op->getSubExpr()->IgnoreParens());
    }
  }
  const std::string Text = getText(SourceManager, ExprNode);
  if (Text.empty()) return std::string();
  // Add leading '*'.
  if (needParensAfterUnaryOperator(ExprNode)) {
    return std::string("*(") + Text + ")";
  }
  return std::string("*") + Text;
}

namespace {
class FixCStrCall : public ast_matchers::MatchFinder::MatchCallback {
 public:
  FixCStrCall(tooling::Replacements *Replace)
      : Replace(Replace) {}

  virtual void run(const ast_matchers::MatchFinder::MatchResult &Result) {
    const CallExpr *Call =
        Result.Nodes.getStmtAs<CallExpr>("call");
    const Expr *Arg =
        Result.Nodes.getStmtAs<Expr>("arg");
    const bool Arrow =
        Result.Nodes.getStmtAs<MemberExpr>("member")->isArrow();
    // Replace the "call" node with the "arg" node, prefixed with '*'
    // if the call was using '->' rather than '.'.
    const std::string ArgText = Arrow ?
        formatDereference(*Result.SourceManager, *Arg) :
        getText(*Result.SourceManager, *Arg);
    if (ArgText.empty()) return;

    Replace->insert(Replacement(*Result.SourceManager, Call, ArgText));
  }

 private:
  tooling::Replacements *Replace;
};
} // end namespace

const char *StringConstructor =
    "::std::basic_string<char, std::char_traits<char>, std::allocator<char> >"
    "::basic_string";

const char *StringCStrMethod =
    "::std::basic_string<char, std::char_traits<char>, std::allocator<char> >"
    "::c_str";

cl::opt<std::string> BuildPath(
  cl::Positional,
  cl::desc("<build-path>"));

cl::list<std::string> SourcePaths(
  cl::Positional,
  cl::desc("<source0> [... <sourceN>]"),
  cl::OneOrMore);

int main(int argc, const char **argv) {
  llvm::OwningPtr<CompilationDatabase> Compilations(
    tooling::FixedCompilationDatabase::loadFromCommandLine(argc, argv));
  cl::ParseCommandLineOptions(argc, argv);
  if (!Compilations) {
    std::string ErrorMessage;
    Compilations.reset(
           CompilationDatabase::loadFromDirectory(BuildPath, ErrorMessage));
    if (!Compilations)
      llvm::report_fatal_error(ErrorMessage);
    }
  tooling::RefactoringTool Tool(*Compilations, SourcePaths);
  ast_matchers::MatchFinder Finder;
  FixCStrCall Callback(&Tool.getReplacements());
  Finder.addMatcher(
      constructExpr(
          hasDeclaration(methodDecl(hasName(StringConstructor))),
          argumentCountIs(2),
          // The first argument must have the form x.c_str() or p->c_str()
          // where the method is string::c_str().  We can use the copy
          // constructor of string instead (or the compiler might share
          // the string object).
          hasArgument(
              0,
              id("call", memberCallExpr(
                  callee(id("member", memberExpr())),
                  callee(methodDecl(hasName(StringCStrMethod))),
                  on(id("arg", expr()))))),
          // The second argument is the alloc object which must not be
          // present explicitly.
          hasArgument(
              1,
              defaultArgExpr())),
      &Callback);
  Finder.addMatcher(
      constructExpr(
          // Implicit constructors of these classes are overloaded
          // wrt. string types and they internally make a StringRef
          // referring to the argument.  Passing a string directly to
          // them is preferred to passing a char pointer.
          hasDeclaration(methodDecl(anyOf(
              hasName("::llvm::StringRef::StringRef"),
              hasName("::llvm::Twine::Twine")))),
          argumentCountIs(1),
          // The only argument must have the form x.c_str() or p->c_str()
          // where the method is string::c_str().  StringRef also has
          // a constructor from string which is more efficient (avoids
          // strlen), so we can construct StringRef from the string
          // directly.
          hasArgument(
              0,
              id("call", memberCallExpr(
                  callee(id("member", memberExpr())),
                  callee(methodDecl(hasName(StringCStrMethod))),
                  on(id("arg", expr())))))),
      &Callback);
  return Tool.run(newFrontendActionFactory(&Finder));
}
