//===--- ASTMatchFinder.h - Structural query framework ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Provides a way to construct an ASTConsumer that runs given matchers
//  over the AST and invokes a given callback on every match.
//
//  The general idea is to construct a matcher expression that describes a
//  subtree match on the AST. Next, a callback that is executed every time the
//  expression matches is registered, and the matcher is run over the AST of
//  some code. Matched subexpressions can be bound to string IDs and easily
//  be accessed from the registered callback. The callback can than use the
//  AST nodes that the subexpressions matched on to output information about
//  the match or construct changes that can be applied to the code.
//
//  Example:
//  class HandleMatch : public MatchFinder::MatchCallback {
//  public:
//    virtual void Run(const MatchFinder::MatchResult &Result) {
//      const CXXRecordDecl *Class =
//          Result.Nodes.GetDeclAs<CXXRecordDecl>("id");
//      ...
//    }
//  };
//
//  int main(int argc, char **argv) {
//    ClangTool Tool(argc, argv);
//    MatchFinder finder;
//    finder.AddMatcher(Id("id", record(hasName("::a_namespace::AClass"))),
//                      new HandleMatch);
//    return Tool.Run(newFrontendActionFactory(&finder));
//  }
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_MATCHERS_AST_MATCH_FINDER_H
#define LLVM_CLANG_AST_MATCHERS_AST_MATCH_FINDER_H

#include "clang/ASTMatchers/ASTMatchers.h"

namespace clang {

namespace ast_matchers {

/// \brief A class to allow finding matches over the Clang AST.
///
/// After creation, you can add multiple matchers to the MatchFinder via
/// calls to addMatcher(...).
///
/// Once all matchers are added, newASTConsumer() returns an ASTConsumer
/// that will trigger the callbacks specified via addMatcher(...) when a match
/// is found.
///
/// See ASTMatchers.h for more information about how to create matchers.
///
/// Not intended to be subclassed.
class MatchFinder {
public:
  /// \brief Contains all information for a given match.
  ///
  /// Every time a match is found, the MatchFinder will invoke the registered
  /// MatchCallback with a MatchResult containing information about the match.
  struct MatchResult {
    MatchResult(const BoundNodes &Nodes, clang::ASTContext *Context);

    /// \brief Contains the nodes bound on the current match.
    ///
    /// This allows user code to easily extract matched AST nodes.
    const BoundNodes Nodes;

    /// \brief Utilities for interpreting the matched AST structures.
    /// @{
    clang::ASTContext * const Context;
    clang::SourceManager * const SourceManager;
    /// @}
  };

  /// \brief Called when the Match registered for it was successfully found
  /// in the AST.
  class MatchCallback {
  public:
    virtual ~MatchCallback();

    /// \brief Called on every match by the \c MatchFinder.
    virtual void run(const MatchResult &Result) = 0;

    /// \brief Called at the start of each translation unit.
    ///
    /// Optionally override to do per translation unit tasks.
    virtual void onStartOfTranslationUnit() {}
  };

  /// \brief Called when parsing is finished. Intended for testing only.
  class ParsingDoneTestCallback {
  public:
    virtual ~ParsingDoneTestCallback();
    virtual void run() = 0;
  };

  MatchFinder();
  ~MatchFinder();

  /// \brief Adds a matcher to execute when running over the AST.
  ///
  /// Calls 'Action' with the BoundNodes on every match.
  /// Adding more than one 'NodeMatch' allows finding different matches in a
  /// single pass over the AST.
  ///
  /// Does not take ownership of 'Action'.
  /// @{
  void addMatcher(const DeclarationMatcher &NodeMatch,
                  MatchCallback *Action);
  void addMatcher(const TypeMatcher &NodeMatch,
                  MatchCallback *Action);
  void addMatcher(const StatementMatcher &NodeMatch,
                  MatchCallback *Action);
  void addMatcher(const NestedNameSpecifierMatcher &NodeMatch,
                  MatchCallback *Action);
  void addMatcher(const NestedNameSpecifierLocMatcher &NodeMatch,
                  MatchCallback *Action);
  void addMatcher(const TypeLocMatcher &NodeMatch,
                  MatchCallback *Action);
  /// @}

  /// \brief Creates a clang ASTConsumer that finds all matches.
  clang::ASTConsumer *newASTConsumer();

  /// \brief Finds all matches on the given \c Node.
  ///
  /// @{
  void findAll(const Decl &Node, ASTContext &Context);
  void findAll(const Stmt &Node, ASTContext &Context);
  /// @}

  /// \brief Registers a callback to notify the end of parsing.
  ///
  /// The provided closure is called after parsing is done, before the AST is
  /// traversed. Useful for benchmarking.
  /// Each call to FindAll(...) will call the closure once.
  void registerTestCallbackAfterParsing(ParsingDoneTestCallback *ParsingDone);

private:
  /// \brief For each \c DynTypedMatcher a \c MatchCallback that will be called
  /// when it matches.
  std::vector<std::pair<const internal::DynTypedMatcher*, MatchCallback*> >
    MatcherCallbackPairs;

  /// \brief Called when parsing is done.
  ParsingDoneTestCallback *ParsingDone;
};

} // end namespace ast_matchers
} // end namespace clang

#endif // LLVM_CLANG_AST_MATCHERS_AST_MATCH_FINDER_H
