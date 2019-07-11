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

  /// Set role for a token starting at \p Loc.
  void markChildToken(SourceLocation Loc, tok::TokenKind Kind, NodeRole R);

  /// Finish building the tree and consume the root node.
  syntax::TranslationUnit *finalize() && {
    auto Tokens = Arena.tokenBuffer().expandedTokens();
    // Build the root of the tree, consuming all the children.
    Pending.foldChildren(Tokens,
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
  llvm::ArrayRef<syntax::Token> getRange(const Stmt *S) const {
    return getRange(S->getBeginLoc(), S->getEndLoc());
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
      // FIXME: do not add 'eof' to the tree.

      // Create all leaf nodes.
      for (auto &T : A.tokenBuffer().expandedTokens())
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

    Builder.markChildToken(S->getLBracLoc(), tok::l_brace,
                           NodeRole::CompoundStatement_lbrace);
    Builder.markChildToken(S->getRBracLoc(), tok::r_brace,
                           NodeRole::CompoundStatement_rbrace);

    Builder.foldNode(Builder.getRange(S),
                     new (allocator()) syntax::CompoundStatement);
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
