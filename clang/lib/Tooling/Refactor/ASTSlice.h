//===--- ASTSlice.h - Represents a portion of the AST ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_TOOLING_REFACTOR_ASTSLICE_H
#define LLVM_CLANG_LIB_TOOLING_REFACTOR_ASTSLICE_H

#include "clang/AST/DeclBase.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {

class NamedDecl;

namespace tooling {

/// Represents a set of statements that overlap with the selection range.
struct SelectedStmtSet {
  /// The outermost statement that contains the start of the selection range.
  const Stmt *containsSelectionRangeStart = nullptr;

  /// The outermost statement that contains the end of the selection range.
  const Stmt *containsSelectionRangeEnd = nullptr;

  /// The innermost statement that contains the entire selection range.
  const Stmt *containsSelectionRange = nullptr;

  /// The index of the innermost statement that contains the entire selection
  /// range. The index points into the NodeTree stored in the \c ASTSlice.
  Optional<unsigned> containsSelectionRangeIndex;

  static SelectedStmtSet createFromEntirelySelected(const Stmt *S,
                                                    unsigned Index);

  /// Returns true if the compound statement is not fully selected.
  bool isCompoundStatementPartiallySelected() const {
    assert(containsSelectionRange && "No statement selected");
    return isa<CompoundStmt>(containsSelectionRange) &&
           (containsSelectionRangeStart || containsSelectionRangeEnd);
  }
};

/// A portion of the AST that is located around the location and/or source
/// range of interest.
class ASTSlice {
public:
  struct Node {
    enum SelectionRangeOverlapKind {
      UnknownOverlap,
      ContainsSelectionRangeStart,
      ContainsSelectionRangeEnd,
      ContainsSelectionRange
    };
    llvm::PointerUnion<const Stmt *, const Decl *> StmtOrDecl;
    const Stmt *ParentStmt;
    const Decl *ParentDecl;
    SourceRange Range;
    SelectionRangeOverlapKind SelectionRangeOverlap = UnknownOverlap;

    const Stmt *getStmtOrNull() const {
      return StmtOrDecl.dyn_cast<const Stmt *>();
    }

    const Decl *getDeclOrNull() const {
      return StmtOrDecl.dyn_cast<const Decl *>();
    }

    Node(const Stmt *S, const Stmt *ParentStmt, const Decl *ParentDecl,
         SourceRange Range)
        : StmtOrDecl(S), ParentStmt(ParentStmt), ParentDecl(ParentDecl),
          Range(Range) {}
    Node(const Decl *D, const Decl *ParentDecl, SourceRange Range)
        : StmtOrDecl(D), ParentStmt(nullptr), ParentDecl(ParentDecl),
          Range(Range) {}
  };

  /// Represents a statement that overlaps with the selection range/point.
  class SelectedStmt {
    ASTSlice &Slice;
    const Stmt *S;
    unsigned Index;

    friend class ASTSlice;

    SelectedStmt(ASTSlice &Slice, const Stmt *S, unsigned Index);

  public:
    const Stmt *getStmt() { return S; }
    const Decl *getParentDecl();
  };

  /// Represents a declaration that overlaps with the selection range/point.
  class SelectedDecl {
    const Decl *D;

    friend class ASTSlice;

    SelectedDecl(const Decl *D);

  public:
    const Decl *getDecl() { return D; }
  };

  ASTSlice(SourceLocation Location, SourceRange SelectionRange,
           ASTContext &Context);

  /// Returns true if the given source range overlaps with the selection.
  bool isSourceRangeSelected(CharSourceRange Range) const;

  enum SelectionSearchOptions {
    /// Search with-in the innermost declaration only, including the declaration
    /// itself without inspecting any other outer declarations.
    InnermostDeclOnly = 1
  };

  /// Returns the statement that results in true when passed into \p Predicate
  /// that's nearest to the location of interest, or \c None if such statement
  /// isn't found.
  Optional<SelectedStmt>
  nearestSelectedStmt(llvm::function_ref<bool(const Stmt *)> Predicate);

  /// Returns the statement of the given class that's nearest to the location
  /// of interest, or \c None if such statement isn't found.
  Optional<SelectedStmt> nearestSelectedStmt(Stmt::StmtClass Class);

  /// TODO: Remove in favour of nearestStmt that returns \c SelectedStmt
  const Stmt *nearestStmt(Stmt::StmtClass Class);

  /// Returns the declaration that overlaps with the selection range, is
  /// nearest to the location of interest and that results in true when passed
  /// into \p Predicate, or \c None if such declaration isn't found.
  Optional<SelectedDecl>
  innermostSelectedDecl(llvm::function_ref<bool(const Decl *)> Predicate,
                        unsigned Options = 0);

  /// Returns the declaration closest to the location of interest whose decl
  /// kind is in \p Classes, or \c None if no such decl can't be found.
  Optional<SelectedDecl> innermostSelectedDecl(ArrayRef<Decl::Kind> Classes,
                                               unsigned Options = 0);

  /// Returns the set of statements that overlap with the selection range.
  Optional<SelectedStmtSet> getSelectedStmtSet();

  /// Returns true if the statement with the given index is contained in a
  /// compound statement that overlaps with the selection range.
  bool isContainedInCompoundStmt(unsigned Index);

  /// Returns the declaration that contains the statement at the given index.
  const Decl *parentDeclForIndex(unsigned Index);

  /// Returns the statement that contains the statement at the given index.
  const Stmt *parentStmtForIndex(unsigned Index);

private:
  Optional<SelectedStmtSet> computeSelectedStmtSet();

  /// Returns the innermost declaration that contains both the start and the
  /// end of the selection range.
  Optional<SelectedDecl> getInnermostCompletelySelectedDecl();

  /// The lowest element is the top of the hierarchy
  SmallVector<Node, 16> NodeTree;
  ASTContext &Context;
  SourceLocation SelectionLocation;
  SourceRange SelectionRange;
  Optional<Optional<SelectedStmtSet>> CachedSelectedStmtSet;
  Optional<Optional<SelectedDecl>> CachedSelectedInnermostDecl;
};

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_LIB_TOOLING_REFACTOR_ASTSLICE_H
