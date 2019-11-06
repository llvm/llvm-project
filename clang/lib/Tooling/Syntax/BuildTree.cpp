//===- BuildTree.cpp ------------------------------------------*- C++ -*-=====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "clang/Tooling/Syntax/BuildTree.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/Syntax/Nodes.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "clang/Tooling/Syntax/Tree.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <map>

using namespace clang;

static bool isImplicitExpr(clang::Expr *E) { return E->IgnoreImplicit() != E; }

/// A helper class for constructing the syntax tree while traversing a clang
/// AST.
///
/// At each point of the traversal we maintain a list of pending nodes.
/// Initially all tokens are added as pending nodes. When processing a clang AST
/// node, the clients need to:
///   - create a corresponding syntax node,
///   - assign roles to all pending child nodes with 'markChild' and
///     'markChildToken',
///   - replace the child nodes with the new syntax node in the pending list
///     with 'foldNode'.
///
/// Note that all children are expected to be processed when building a node.
///
/// Call finalize() to finish building the tree and consume the root node.
class syntax::TreeBuilder {
public:
  TreeBuilder(syntax::Arena &Arena) : Arena(Arena), Pending(Arena) {}

  llvm::BumpPtrAllocator &allocator() { return Arena.allocator(); }

  /// Populate children for \p New node, assuming it covers tokens from \p
  /// Range.
  void foldNode(llvm::ArrayRef<syntax::Token> Range, syntax::Tree *New);

  /// Mark the \p Child node with a corresponding \p Role. All marked children
  /// should be consumed by foldNode.
  /// (!) when called on expressions (clang::Expr is derived from clang::Stmt),
  ///     wraps expressions into expression statement.
  void markStmtChild(Stmt *Child, NodeRole Role);
  /// Should be called for expressions in non-statement position to avoid
  /// wrapping into expression statement.
  void markExprChild(Expr *Child, NodeRole Role);

  /// Set role for a token starting at \p Loc.
  void markChildToken(SourceLocation Loc, tok::TokenKind Kind, NodeRole R);

  /// Finish building the tree and consume the root node.
  syntax::TranslationUnit *finalize() && {
    auto Tokens = Arena.tokenBuffer().expandedTokens();
    assert(!Tokens.empty());
    assert(Tokens.back().kind() == tok::eof);

    // Build the root of the tree, consuming all the children.
    Pending.foldChildren(Tokens.drop_back(),
                         new (Arena.allocator()) syntax::TranslationUnit);

    return cast<syntax::TranslationUnit>(std::move(Pending).finalize());
  }

  /// getRange() finds the syntax tokens corresponding to the passed source
  /// locations.
  /// \p First is the start position of the first token and \p Last is the start
  /// position of the last token.
  llvm::ArrayRef<syntax::Token> getRange(SourceLocation First,
                                         SourceLocation Last) const {
    assert(First.isValid());
    assert(Last.isValid());
    assert(First == Last ||
           Arena.sourceManager().isBeforeInTranslationUnit(First, Last));
    return llvm::makeArrayRef(findToken(First), std::next(findToken(Last)));
  }
  llvm::ArrayRef<syntax::Token> getRange(const Decl *D) const {
    return getRange(D->getBeginLoc(), D->getEndLoc());
  }
  llvm::ArrayRef<syntax::Token> getExprRange(const Expr *E) const {
    return getRange(E->getBeginLoc(), E->getEndLoc());
  }
  /// Find the adjusted range for the statement, consuming the trailing
  /// semicolon when needed.
  llvm::ArrayRef<syntax::Token> getStmtRange(const Stmt *S) const {
    auto Tokens = getRange(S->getBeginLoc(), S->getEndLoc());
    if (isa<CompoundStmt>(S))
      return Tokens;

    // Some statements miss a trailing semicolon, e.g. 'return', 'continue' and
    // all statements that end with those. Consume this semicolon here.
    //
    // (!) statements never consume 'eof', so looking at the next token is ok.
    if (Tokens.back().kind() != tok::semi && Tokens.end()->kind() == tok::semi)
      return llvm::makeArrayRef(Tokens.begin(), Tokens.end() + 1);
    return Tokens;
  }

private:
  /// Finds a token starting at \p L. The token must exist.
  const syntax::Token *findToken(SourceLocation L) const;

  /// A collection of trees covering the input tokens.
  /// When created, each tree corresponds to a single token in the file.
  /// Clients call 'foldChildren' to attach one or more subtrees to a parent
  /// node and update the list of trees accordingly.
  ///
  /// Ensures that added nodes properly nest and cover the whole token stream.
  struct Forest {
    Forest(syntax::Arena &A) {
      assert(!A.tokenBuffer().expandedTokens().empty());
      assert(A.tokenBuffer().expandedTokens().back().kind() == tok::eof);
      // Create all leaf nodes.
      // Note that we do not have 'eof' in the tree.
      for (auto &T : A.tokenBuffer().expandedTokens().drop_back())
        Trees.insert(Trees.end(),
                     {&T, NodeAndRole{new (A.allocator()) syntax::Leaf(&T)}});
    }

    void assignRole(llvm::ArrayRef<syntax::Token> Range,
                    syntax::NodeRole Role) {
      assert(!Range.empty());
      auto It = Trees.lower_bound(Range.begin());
      assert(It != Trees.end() && "no node found");
      assert(It->first == Range.begin() && "no child with the specified range");
      assert((std::next(It) == Trees.end() ||
              std::next(It)->first == Range.end()) &&
             "no child with the specified range");
      It->second.Role = Role;
    }

    /// Add \p Node to the forest and fill its children nodes based on the \p
    /// NodeRange.
    void foldChildren(llvm::ArrayRef<syntax::Token> NodeTokens,
                      syntax::Tree *Node) {
      assert(!NodeTokens.empty());
      assert(Node->firstChild() == nullptr && "node already has children");

      auto *FirstToken = NodeTokens.begin();
      auto BeginChildren = Trees.lower_bound(FirstToken);
      assert(BeginChildren != Trees.end() &&
             BeginChildren->first == FirstToken &&
             "fold crosses boundaries of existing subtrees");
      auto EndChildren = Trees.lower_bound(NodeTokens.end());
      assert((EndChildren == Trees.end() ||
              EndChildren->first == NodeTokens.end()) &&
             "fold crosses boundaries of existing subtrees");

      // (!) we need to go in reverse order, because we can only prepend.
      for (auto It = EndChildren; It != BeginChildren; --It)
        Node->prependChildLowLevel(std::prev(It)->second.Node,
                                   std::prev(It)->second.Role);

      Trees.erase(BeginChildren, EndChildren);
      Trees.insert({FirstToken, NodeAndRole(Node)});
    }

    // EXPECTS: all tokens were consumed and are owned by a single root node.
    syntax::Node *finalize() && {
      assert(Trees.size() == 1);
      auto *Root = Trees.begin()->second.Node;
      Trees = {};
      return Root;
    }

    std::string str(const syntax::Arena &A) const {
      std::string R;
      for (auto It = Trees.begin(); It != Trees.end(); ++It) {
        unsigned CoveredTokens =
            It != Trees.end()
                ? (std::next(It)->first - It->first)
                : A.tokenBuffer().expandedTokens().end() - It->first;

        R += llvm::formatv("- '{0}' covers '{1}'+{2} tokens\n",
                           It->second.Node->kind(),
                           It->first->text(A.sourceManager()), CoveredTokens);
        R += It->second.Node->dump(A);
      }
      return R;
    }

  private:
    /// A with a role that should be assigned to it when adding to a parent.
    struct NodeAndRole {
      explicit NodeAndRole(syntax::Node *Node)
          : Node(Node), Role(NodeRole::Unknown) {}

      syntax::Node *Node;
      NodeRole Role;
    };

    /// Maps from the start token to a subtree starting at that token.
    /// FIXME: storing the end tokens is redundant.
    /// FIXME: the key of a map is redundant, it is also stored in NodeForRange.
    std::map<const syntax::Token *, NodeAndRole> Trees;
  };

  /// For debugging purposes.
  std::string str() { return Pending.str(Arena); }

  syntax::Arena &Arena;
  Forest Pending;
};

namespace {
class BuildTreeVisitor : public RecursiveASTVisitor<BuildTreeVisitor> {
public:
  explicit BuildTreeVisitor(ASTContext &Ctx, syntax::TreeBuilder &Builder)
      : Builder(Builder), LangOpts(Ctx.getLangOpts()) {}

  bool shouldTraversePostOrder() const { return true; }

  bool TraverseDecl(Decl *D) {
    if (!D || isa<TranslationUnitDecl>(D))
      return RecursiveASTVisitor::TraverseDecl(D);
    if (!llvm::isa<TranslationUnitDecl>(D->getDeclContext()))
      return true; // Only build top-level decls for now, do not recurse.
    return RecursiveASTVisitor::TraverseDecl(D);
  }

  bool VisitDecl(Decl *D) {
    assert(llvm::isa<TranslationUnitDecl>(D->getDeclContext()) &&
           "expected a top-level decl");
    assert(!D->isImplicit());
    Builder.foldNode(Builder.getRange(D),
                     new (allocator()) syntax::TopLevelDeclaration());
    return true;
  }

  bool WalkUpFromTranslationUnitDecl(TranslationUnitDecl *TU) {
    // (!) we do not want to call VisitDecl(), the declaration for translation
    // unit is built by finalize().
    return true;
  }

  bool WalkUpFromCompoundStmt(CompoundStmt *S) {
    using NodeRole = syntax::NodeRole;

    Builder.markChildToken(S->getLBracLoc(), tok::l_brace, NodeRole::OpenParen);
    for (auto *Child : S->body())
      Builder.markStmtChild(Child, NodeRole::CompoundStatement_statement);
    Builder.markChildToken(S->getRBracLoc(), tok::r_brace,
                           NodeRole::CloseParen);

    Builder.foldNode(Builder.getStmtRange(S),
                     new (allocator()) syntax::CompoundStatement);
    return true;
  }

  // Some statements are not yet handled by syntax trees.
  bool WalkUpFromStmt(Stmt *S) {
    Builder.foldNode(Builder.getStmtRange(S),
                     new (allocator()) syntax::UnknownStatement);
    return true;
  }

  bool TraverseCXXForRangeStmt(CXXForRangeStmt *S) {
    // We override to traverse range initializer as VarDecl.
    // RAV traverses it as a statement, we produce invalid node kinds in that
    // case.
    // FIXME: should do this in RAV instead?
    if (S->getInit() && !TraverseStmt(S->getInit()))
      return false;
    if (S->getLoopVariable() && !TraverseDecl(S->getLoopVariable()))
      return false;
    if (S->getRangeInit() && !TraverseStmt(S->getRangeInit()))
      return false;
    if (S->getBody() && !TraverseStmt(S->getBody()))
      return false;
    return true;
  }

  bool TraverseStmt(Stmt *S) {
    if (auto *E = llvm::dyn_cast_or_null<Expr>(S)) {
      // (!) do not recurse into subexpressions.
      // we do not have syntax trees for expressions yet, so we only want to see
      // the first top-level expression.
      return WalkUpFromExpr(E->IgnoreImplicit());
    }
    return RecursiveASTVisitor::TraverseStmt(S);
  }

  // Some expressions are not yet handled by syntax trees.
  bool WalkUpFromExpr(Expr *E) {
    assert(!isImplicitExpr(E) && "should be handled by TraverseStmt");
    Builder.foldNode(Builder.getExprRange(E),
                     new (allocator()) syntax::UnknownExpression);
    return true;
  }

  // The code below is very regular, it could even be generated with some
  // preprocessor magic. We merely assign roles to the corresponding children
  // and fold resulting nodes.
  bool WalkUpFromDeclStmt(DeclStmt *S) {
    Builder.foldNode(Builder.getStmtRange(S),
                     new (allocator()) syntax::DeclarationStatement);
    return true;
  }

  bool WalkUpFromNullStmt(NullStmt *S) {
    Builder.foldNode(Builder.getStmtRange(S),
                     new (allocator()) syntax::EmptyStatement);
    return true;
  }

  bool WalkUpFromSwitchStmt(SwitchStmt *S) {
    Builder.markChildToken(S->getSwitchLoc(), tok::kw_switch,
                           syntax::NodeRole::IntroducerKeyword);
    Builder.markStmtChild(S->getBody(), syntax::NodeRole::BodyStatement);
    Builder.foldNode(Builder.getStmtRange(S),
                     new (allocator()) syntax::SwitchStatement);
    return true;
  }

  bool WalkUpFromCaseStmt(CaseStmt *S) {
    Builder.markChildToken(S->getKeywordLoc(), tok::kw_case,
                           syntax::NodeRole::IntroducerKeyword);
    Builder.markExprChild(S->getLHS(), syntax::NodeRole::CaseStatement_value);
    Builder.markStmtChild(S->getSubStmt(), syntax::NodeRole::BodyStatement);
    Builder.foldNode(Builder.getStmtRange(S),
                     new (allocator()) syntax::CaseStatement);
    return true;
  }

  bool WalkUpFromDefaultStmt(DefaultStmt *S) {
    Builder.markChildToken(S->getKeywordLoc(), tok::kw_default,
                           syntax::NodeRole::IntroducerKeyword);
    Builder.markStmtChild(S->getSubStmt(), syntax::NodeRole::BodyStatement);
    Builder.foldNode(Builder.getStmtRange(S),
                     new (allocator()) syntax::DefaultStatement);
    return true;
  }

  bool WalkUpFromIfStmt(IfStmt *S) {
    Builder.markChildToken(S->getIfLoc(), tok::kw_if,
                           syntax::NodeRole::IntroducerKeyword);
    Builder.markStmtChild(S->getThen(),
                          syntax::NodeRole::IfStatement_thenStatement);
    Builder.markChildToken(S->getElseLoc(), tok::kw_else,
                           syntax::NodeRole::IfStatement_elseKeyword);
    Builder.markStmtChild(S->getElse(),
                          syntax::NodeRole::IfStatement_elseStatement);
    Builder.foldNode(Builder.getStmtRange(S),
                     new (allocator()) syntax::IfStatement);
    return true;
  }

  bool WalkUpFromForStmt(ForStmt *S) {
    Builder.markChildToken(S->getForLoc(), tok::kw_for,
                           syntax::NodeRole::IntroducerKeyword);
    Builder.markStmtChild(S->getBody(), syntax::NodeRole::BodyStatement);
    Builder.foldNode(Builder.getStmtRange(S),
                     new (allocator()) syntax::ForStatement);
    return true;
  }

  bool WalkUpFromWhileStmt(WhileStmt *S) {
    Builder.markChildToken(S->getWhileLoc(), tok::kw_while,
                           syntax::NodeRole::IntroducerKeyword);
    Builder.markStmtChild(S->getBody(), syntax::NodeRole::BodyStatement);
    Builder.foldNode(Builder.getStmtRange(S),
                     new (allocator()) syntax::WhileStatement);
    return true;
  }

  bool WalkUpFromContinueStmt(ContinueStmt *S) {
    Builder.markChildToken(S->getContinueLoc(), tok::kw_continue,
                           syntax::NodeRole::IntroducerKeyword);
    Builder.foldNode(Builder.getStmtRange(S),
                     new (allocator()) syntax::ContinueStatement);
    return true;
  }

  bool WalkUpFromBreakStmt(BreakStmt *S) {
    Builder.markChildToken(S->getBreakLoc(), tok::kw_break,
                           syntax::NodeRole::IntroducerKeyword);
    Builder.foldNode(Builder.getStmtRange(S),
                     new (allocator()) syntax::BreakStatement);
    return true;
  }

  bool WalkUpFromReturnStmt(ReturnStmt *S) {
    Builder.markChildToken(S->getReturnLoc(), tok::kw_return,
                           syntax::NodeRole::IntroducerKeyword);
    Builder.markExprChild(S->getRetValue(),
                          syntax::NodeRole::ReturnStatement_value);
    Builder.foldNode(Builder.getStmtRange(S),
                     new (allocator()) syntax::ReturnStatement);
    return true;
  }

  bool WalkUpFromCXXForRangeStmt(CXXForRangeStmt *S) {
    Builder.markChildToken(S->getForLoc(), tok::kw_for,
                           syntax::NodeRole::IntroducerKeyword);
    Builder.markStmtChild(S->getBody(), syntax::NodeRole::BodyStatement);
    Builder.foldNode(Builder.getStmtRange(S),
                     new (allocator()) syntax::RangeBasedForStatement);
    return true;
  }

private:
  /// A small helper to save some typing.
  llvm::BumpPtrAllocator &allocator() { return Builder.allocator(); }

  syntax::TreeBuilder &Builder;
  const LangOptions &LangOpts;
};
} // namespace

void syntax::TreeBuilder::foldNode(llvm::ArrayRef<syntax::Token> Range,
                                   syntax::Tree *New) {
  Pending.foldChildren(Range, New);
}

void syntax::TreeBuilder::markChildToken(SourceLocation Loc,
                                         tok::TokenKind Kind, NodeRole Role) {
  if (Loc.isInvalid())
    return;
  Pending.assignRole(*findToken(Loc), Role);
}

void syntax::TreeBuilder::markStmtChild(Stmt *Child, NodeRole Role) {
  if (!Child)
    return;

  auto Range = getStmtRange(Child);
  // This is an expression in a statement position, consume the trailing
  // semicolon and form an 'ExpressionStatement' node.
  if (auto *E = dyn_cast<Expr>(Child)) {
    Pending.assignRole(getExprRange(E),
                       NodeRole::ExpressionStatement_expression);
    // (!) 'getRange(Stmt)' ensures this already covers a trailing semicolon.
    Pending.foldChildren(Range, new (allocator()) syntax::ExpressionStatement);
  }
  Pending.assignRole(Range, Role);
}

void syntax::TreeBuilder::markExprChild(Expr *Child, NodeRole Role) {
  Pending.assignRole(getExprRange(Child), Role);
}

const syntax::Token *syntax::TreeBuilder::findToken(SourceLocation L) const {
  auto Tokens = Arena.tokenBuffer().expandedTokens();
  auto &SM = Arena.sourceManager();
  auto It = llvm::partition_point(Tokens, [&](const syntax::Token &T) {
    return SM.isBeforeInTranslationUnit(T.location(), L);
  });
  assert(It != Tokens.end());
  assert(It->location() == L);
  return &*It;
}

syntax::TranslationUnit *
syntax::buildSyntaxTree(Arena &A, const TranslationUnitDecl &TU) {
  TreeBuilder Builder(A);
  BuildTreeVisitor(TU.getASTContext(), Builder).TraverseAST(TU.getASTContext());
  return std::move(Builder).finalize();
}
