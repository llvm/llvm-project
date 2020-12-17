//===--- Selection.h - What's under the cursor? -------------------*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Many features are triggered at locations/ranges and operate on AST nodes.
// (e.g. go-to-definition or code tweaks).
// At a high level, such features need to work out which node is the correct
// target.
//
// There are a few levels of ambiguity here:
//
// Which tokens are included:
//   int x = one + two;  // what should "go to definition" do?
//            ^^^^^^
//
// Same token means multiple things:
//   string("foo")       // class string, or a constructor?
//   ^
//
// Which level of the AST is interesting?
//   if (err) {          // reference to 'err', or operator bool(),
//       ^               // or the if statement itself?
//
// Here we build and expose a data structure that allows features to resolve
// these ambiguities in an appropriate way:
//   - we determine which low-level nodes are partly or completely covered
//     by the selection.
//   - we expose a tree of the selected nodes and their lexical parents.
//
// Sadly LSP specifies locations as being between characters, and this causes
// some ambiguities we cannot cleanly resolve:
//   lhs+rhs  // targeting '+' or 'lhs'?
//      ^     // in GUI editors, double-clicking 'lhs' yields this position!
//
// The best we can do in these cases is try both, which leads to the awkward
// SelectionTree::createEach() API.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_SELECTION_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_SELECTION_H
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {
namespace clangd {

// A selection can partially or completely cover several AST nodes.
// The SelectionTree contains nodes that are covered, and their parents.
// SelectionTree does not contain all AST nodes, rather only:
//   Decl, Stmt, TypeLoc, NestedNamespaceSpecifierLoc, CXXCtorInitializer.
// (These are the nodes with source ranges that fit in DynTypedNode).
//
// Usually commonAncestor() is the place to start:
//  - it's the simplest answer to "what node is under the cursor"
//  - the selected Expr (for example) can be found by walking up the parent
//    chain and checking Node->ASTNode.
//  - if you want to traverse the selected nodes, they are all under
//    commonAncestor() in the tree.
//
// SelectionTree tries to behave sensibly in the presence of macros, but does
// not model any preprocessor concepts: the output is a subset of the AST.
// When a macro argument is specifically selected, only its first expansion is
// selected in the AST. (Returning a selection forest is unreasonably difficult
// for callers to handle correctly.)
//
// Comments, directives and whitespace are completely ignored.
// Semicolons are also ignored, as the AST generally does not model them well.
//
// The SelectionTree owns the Node structures, but the ASTNode attributes
// point back into the AST it was constructed with.
class SelectionTree {
public:
  // Create selection trees for the given range, and pass them to Func.
  //
  // There may be multiple possible selection trees:
  // - if the range is empty and borders two tokens, a tree for the right token
  //   and a tree for the left token will be yielded.
  // - Func should return true on success (stop) and false on failure (continue)
  //
  // Always yields at least one tree. If no tokens are touched, it is empty.
  static bool createEach(ASTContext &AST, const syntax::TokenBuffer &Tokens,
                         unsigned Begin, unsigned End,
                         llvm::function_ref<bool(SelectionTree)> Func);

  // Create a selection tree for the given range.
  //
  // Where ambiguous (range is empty and borders two tokens), prefer the token
  // on the right.
  static SelectionTree createRight(ASTContext &AST,
                                   const syntax::TokenBuffer &Tokens,
                                   unsigned Begin, unsigned End);

  // Copies are no good - contain pointers to other nodes.
  SelectionTree(const SelectionTree &) = delete;
  SelectionTree &operator=(const SelectionTree &) = delete;
  // Moves are OK though - internal storage is pointer-stable when moved.
  SelectionTree(SelectionTree &&) = default;
  SelectionTree &operator=(SelectionTree &&) = default;

  // Describes to what extent an AST node is covered by the selection.
  enum Selection : unsigned char {
    // The AST node owns no characters covered by the selection.
    // Note that characters owned by children don't count:
    //   if (x == 0) scream();
    //       ^^^^^^
    // The IfStmt would be Unselected because all the selected characters are
    // associated with its children.
    // (Invisible nodes like ImplicitCastExpr are always unselected).
    Unselected,
    // The AST node owns selected characters, but is not completely covered.
    Partial,
    // The AST node owns characters, and is covered by the selection.
    Complete,
  };
  // An AST node that is implicated in the selection.
  // (Either selected directly, or some descendant is selected).
  struct Node {
    // The parent within the selection tree. nullptr for TranslationUnitDecl.
    Node *Parent;
    // Direct children within the selection tree.
    llvm::SmallVector<const Node *> Children;
    // The corresponding node from the full AST.
    DynTypedNode ASTNode;
    // The extent to which this node is covered by the selection.
    Selection Selected;
    // Walk up the AST to get the DeclContext of this Node,
    // which is not the node itself.
    const DeclContext &getDeclContext() const;
    // Printable node kind, like "CXXRecordDecl" or "AutoTypeLoc".
    std::string kind() const;
    // If this node is a wrapper with no syntax (e.g. implicit cast), return
    // its contents. (If multiple wrappers are present, unwraps all of them).
    const Node &ignoreImplicit() const;
    // If this node is inside a wrapper with no syntax (e.g. implicit cast),
    // return that wrapper. (If multiple are present, unwraps all of them).
    const Node &outerImplicit() const;
  };
  // The most specific common ancestor of all the selected nodes.
  // Returns nullptr if the common ancestor is the root.
  // (This is to avoid accidentally traversing the TUDecl and thus preamble).
  const Node *commonAncestor() const;
  // The selection node corresponding to TranslationUnitDecl.
  const Node &root() const { return *Root; }

private:
  // Creates a selection tree for the given range in the main file.
  // The range includes bytes [Start, End).
  SelectionTree(ASTContext &AST, const syntax::TokenBuffer &Tokens,
                unsigned Start, unsigned End);

  std::deque<Node> Nodes; // Stable-pointer storage.
  const Node *Root;
  clang::PrintingPolicy PrintPolicy;

  void print(llvm::raw_ostream &OS, const Node &N, int Indent) const;
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                       const SelectionTree &T) {
    T.print(OS, T.root(), 1);
    return OS;
  }
};

} // namespace clangd
} // namespace clang
#endif
