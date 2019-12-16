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
//
// Comments, directives and whitespace are completely ignored.
// Semicolons are also ignored, as the AST generally does not model them well.
//
// The SelectionTree owns the Node structures, but the ASTNode attributes
// point back into the AST it was constructed with.
class SelectionTree {
public:
  // Creates a selection tree at the given byte offset in the main file.
  // This is approximately equivalent to a range of one character.
  // (Usually, the character to the right of Offset, sometimes to the left).
  SelectionTree(ASTContext &AST, const syntax::TokenBuffer &Tokens,
                unsigned Offset);
  // Creates a selection tree for the given range in the main file.
  // The range includes bytes [Start, End).
  // If Start == End, uses the same heuristics as SelectionTree(AST, Start).
  SelectionTree(ASTContext &AST, const syntax::TokenBuffer &Tokens,
                unsigned Start, unsigned End);

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
    llvm::SmallVector<const Node *, 8> Children;
    // The corresponding node from the full AST.
    ast_type_traits::DynTypedNode ASTNode;
    // The extent to which this node is covered by the selection.
    Selection Selected;
    // Walk up the AST to get the DeclContext of this Node,
    // which is not the node itself.
    const DeclContext& getDeclContext() const;
    // Printable node kind, like "CXXRecordDecl" or "AutoTypeLoc".
    std::string kind() const;
    // If this node is a wrapper with no syntax (e.g. implicit cast), return
    // its contents. (If multiple wrappers are present, unwraps all of them).
    const Node& ignoreImplicit() const;
    // If this node is inside a wrapper with no syntax (e.g. implicit cast),
    // return that wrapper. (If multiple are present, unwraps all of them).
    const Node& outerImplicit() const;
  };
  // The most specific common ancestor of all the selected nodes.
  // Returns nullptr if the common ancestor is the root.
  // (This is to avoid accidentally traversing the TUDecl and thus preamble).
  const Node *commonAncestor() const;
  // The selection node corresponding to TranslationUnitDecl.
  const Node &root() const { return *Root; }

private:
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
