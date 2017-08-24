//===--- ASTSelection.h - Clang refactoring library -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTOR_AST_SELECTION_H
#define LLVM_CLANG_TOOLING_REFACTOR_AST_SELECTION_H

#include "clang/AST/ASTTypeTraits.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include <vector>

namespace clang {

class ASTContext;

namespace tooling {

enum class SourceSelectionKind {
  /// A node that's not selected.
  None,

  /// A node that's considered to be selected because the whole selection range
  /// is inside of its source range.
  ContainsSelection,
  /// A node that's considered to be selected because the start of the selection
  /// range is inside its source range.
  ContainsSelectionStart,
  /// A node that's considered to be selected because the end of the selection
  /// range is inside its source range.
  ContainsSelectionEnd,

  /// A node that's considered to be selected because the node is entirely in
  /// the selection range.
  InsideSelection,
};

/// Represents a selected AST node.
///
/// AST selection is represented using a tree of \c SelectedASTNode. The tree
/// follows the top-down shape of the actual AST. Each selected node has
/// a selection kind. The kind might be none as the node itself might not
/// actually be selected, e.g. a statement in macro whose child is in a macro
/// argument.
struct SelectedASTNode {
  ast_type_traits::DynTypedNode Node;
  SourceSelectionKind SelectionKind;
  std::vector<SelectedASTNode> Children;

  SelectedASTNode(const ast_type_traits::DynTypedNode &Node,
                  SourceSelectionKind SelectionKind)
      : Node(Node), SelectionKind(SelectionKind) {}
  SelectedASTNode(SelectedASTNode &&) = default;
  SelectedASTNode &operator=(SelectedASTNode &&) = default;

  void dump(llvm::raw_ostream &OS = llvm::errs()) const;
};

/// Traverses the given ASTContext and creates a tree of selected AST nodes.
///
/// \returns None if no nodes are selected in the AST, or a selected AST node
/// that corresponds to the TranslationUnitDecl otherwise.
Optional<SelectedASTNode> findSelectedASTNodes(const ASTContext &Context,
                                               SourceRange SelectionRange);

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_REFACTOR_AST_SELECTION_H
