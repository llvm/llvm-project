//===- Origins.h - Origin and Origin Management ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines Origins, which represent the set of possible loans a
// pointer-like object could hold, and the OriginManager, which manages the
// creation, storage, and retrieval of origins for variables and expressions.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_ORIGINS_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_ORIGINS_H

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/TypeBase.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LifetimeStats.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Utils.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

namespace clang::lifetimes::internal {

using OriginID = utils::ID<struct OriginTag>;

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, OriginID ID) {
  return OS << ID.Value;
}

/// An Origin is a symbolic identifier that represents the set of possible
/// loans a pointer-like object could hold at any given time.
///
/// Each Origin corresponds to a single level of indirection. For complex types
/// with multiple levels of indirection (e.g., `int**`), multiple Origins are
/// organized into a tree structure (see below).
struct Origin {
  OriginID ID;
  /// A pointer to the AST node that this origin represents. This union
  /// distinguishes between origins from declarations (variables or parameters)
  /// and origins from expressions.
  llvm::PointerUnion<const clang::ValueDecl *, const clang::Expr *> Ptr;

  /// The type at this indirection level.
  ///
  /// For `int** pp`:
  ///   Root origin: QT = `int**` (what pp points to)
  ///   Pointee origin: QT = `int*` (what *pp points to)
  ///
  /// Null for synthetic lvalue origins (e.g., outer origin of DeclRefExpr).
  const Type *Ty;

  Origin(OriginID ID, const clang::ValueDecl *D, const Type *QT)
      : ID(ID), Ptr(D), Ty(QT) {}
  Origin(OriginID ID, const clang::Expr *E, const Type *QT)
      : ID(ID), Ptr(E), Ty(QT) {}

  const clang::ValueDecl *getDecl() const {
    return Ptr.dyn_cast<const clang::ValueDecl *>();
  }
  const clang::Expr *getExpr() const {
    return Ptr.dyn_cast<const clang::Expr *>();
  }
};

/// A tree of origins representing the structure of a pointer-like or
/// record type.
///
/// Each node carries an OriginID and is connected to children via labeled
/// edges: either a pointee edge (one level of pointer/reference indirection)
/// or a field edge (a named field of a record). Pointer-like types form a
/// pointee chain; record types fan out via field edges.
///
/// Examples:
///   - For `int& x`, a single origin: what `x` refers to.
///
///   - For `int* p`, the chain has length 2:
///       Outer: the pointer variable `p`
///        +-pointee-> Inner: what `p` points to
///
///   - For `View v` (View is gsl::Pointer), the chain has length 2:
///       Outer: the view object itself
///        +-pointee-> Inner: what the view refers to
///
///   - For `int** pp`, the chain has length 3:
///       Outer: `pp` itself
///        +-pointee-> Inner: `*pp` (what `pp` points to)
///           +-pointee-> Inner->Inner: `**pp` (what `*pp` points to)
///
///   - For `struct S { View a; Inner b; }` (with `struct Inner { View c; }`),
///     the node fans out into a tree, one `field` edge per field with origins:
///       O_s: the record `s` (holds no loans directly)
///        +-field a-> O_a: what the `View` field `s.a` refers to
///        +-field b-> O_b: the record `s.b` (holds no loans directly)
///           +-field c-> O_c: what the `View` field `s.b.c` refers to
///
/// The structure enables the analysis to track how loans flow through
/// levels of indirection and across record fields when assignments and
/// dereferences occur.
class OriginNode {
public:
  /// A labeled edge from this node to a child. The `FD` label determines the
  /// edge type:
  ///   - null `FD`: a pointee edge (one level of pointer/reference indirection)
  ///   - non-null `FD`: a field edge (the named field of a record type)
  ///
  /// The label allows the same child subtree to be reachable via different
  /// relationships. For example, the subtree for field `v` in `s.v` can be
  /// reached both
  ///   (1) as a field child from `s`'s node (with FD=v), and
  ///   (2) as a pointee child from the lvalue node for `s.v` (with FD=null).
  struct Edge {
    const FieldDecl *FD;
    OriginNode *Child;
  };

  OriginNode(OriginID OID) : OID(OID) {}

  OriginID getOriginID() const { return OID; }

  llvm::ArrayRef<Edge> children() const { return Children; }

  template <typename Fn> void forEachOrigin(Fn F) const {
    llvm::SmallVector<const OriginNode *, 4> Worklist{this};
    while (!Worklist.empty()) {
      const OriginNode *N = Worklist.pop_back_val();
      F(N);
      for (const Edge &E : N->children())
        Worklist.push_back(E.Child);
    }
  }

  OriginNode *getPointeeChild() const {
    for (const Edge &E : Children)
      if (!E.FD)
        return E.Child;
    return nullptr;
  }

  OriginNode *getFieldChild(const FieldDecl &F) const {
    for (const Edge &E : Children)
      if (E.FD == &F)
        return E.Child;
    return nullptr;
  }

  /// To reach the record, peels the base's outer origin when the
  /// base is a glvalue (`IsGLValue`) and one more level for an arrow access
  /// (`IsArrow`), then looks up `FD`. Returns null if `FD` is not reachable.
  OriginNode *resolveMemberField(const FieldDecl *FD, bool IsGLValue,
                                 bool IsArrow) const {
    assert(FD);
    const OriginNode *N = this;
    if (IsGLValue)
      N = N->getPointeeChild();
    if (IsArrow && N)
      N = N->getPointeeChild();
    return N ? N->getFieldChild(FD) : nullptr;
  }

  void setChildren(llvm::ArrayRef<Edge> NewChildren) {
    assert(Children.empty() && "children must be set at most once");
    Children = NewChildren;
  }

  // Used to compare two chains' lengths.
  size_t getPointeeChainLength() const {
    size_t Length = 1;
    const OriginNode *T = this;
    while (auto *ON = T->getPointeeChild()) {
      T = ON;
      Length++;
    }
    return Length;
  }

private:
  OriginID OID;
  llvm::ArrayRef<Edge> Children;
};

bool doesDeclHaveStorage(const ValueDecl *D);

/// Manages the creation, storage, and retrieval of origins for pointer-like
/// variables and expressions.
class OriginManager {
public:
  explicit OriginManager(const AnalysisDeclContext &AC);

  /// Gets or creates the OriginNode for a given ValueDecl.
  ///
  /// Creates a tree structure mirroring the levels of indirection in the
  /// declaration's type (e.g., `int* p` creates a chain of length 2).
  ///
  /// \returns The OriginNode, or nullptr if the type is not pointer-like.
  OriginNode *getOrCreateNode(const ValueDecl *D);

  /// Gets or creates the OriginNode for a given Expr.
  ///
  /// Creates a tree structure based on the expression's type and value
  /// category:
  /// - Lvalues get an implicit reference level (modeling addressability)
  /// - Rvalues of non-pointer type return nullptr (no trackable origin)
  /// - DeclRefExpr may reuse the underlying declaration's tree
  ///
  /// \returns The OriginNode, or nullptr for non-pointer rvalues.
  OriginNode *getOrCreateNode(const Expr *E);

  /// Wraps an existing OriginID in a new single-element OriginNode, so a fact
  /// can refer to a single level of an existing OriginNode.
  OriginNode *createSingleOriginNode(OriginID OID);

  /// Returns the OriginNode for the implicit 'this' parameter if the current
  /// declaration is an instance method.
  std::optional<OriginNode *> getThisOrigins() const { return ThisOrigins; }

  const Origin &getOrigin(OriginID ID) const;

  llvm::ArrayRef<Origin> getOrigins() const { return AllOrigins; }

  unsigned getNumOrigins() const { return NextOriginID.Value; }

  bool hasOrigins(QualType QT) const;
  bool hasOrigins(const Expr *E) const;

  bool isAccessedField(const FieldDecl *FD) const {
    return AccessedFields.contains(FD);
  }

  void dump(OriginID OID, llvm::raw_ostream &OS,
            const FieldDecl *FD = nullptr) const;

  /// Collects statistics about expressions that lack associated origins.
  void collectMissingOrigins(Stmt &FunctionBody, LifetimeSafetyStats &LSStats);

private:
  OriginID getNextOriginID() { return NextOriginID++; }

  OriginNode *createNode(const ValueDecl *D, QualType QT);
  OriginNode *createNode(const Expr *E, QualType QT);

  void attachPointeeChild(OriginNode *Parent, OriginNode *Pointee);
  void attachChildren(OriginNode *Parent,
                      llvm::ArrayRef<OriginNode::Edge> Children);

  template <typename T>
  OriginNode *buildNodeForType(QualType QT, const T *Node);
  template <typename T>
  OriginNode *buildNodeForTypeImpl(QualType QT, const T *Node,
                                   llvm::SmallPtrSet<const Type *, 4> &Visited,
                                   unsigned FieldDepth);

  /// Whether a record field participates in origin tracking.
  bool isTrackedField(const FieldDecl *FD) const;

  void initializeThisOrigins(const Decl *D);

  /// Pre-scans the function body (and constructor init lists) to discover:
  ///
  /// 1. Return types of lifetime-annotated calls (currently
  ///    [[clang::lifetimebound]]), registering them for origin tracking.
  ///
  /// 2. The fields it accesses; the rest are excluded from origin tracking.
  void runPreScan(const AnalysisDeclContext &AC);
  void registerLifetimeAnnotatedOriginType(QualType QT);

  ASTContext &AST;
  OriginID NextOriginID{0};
  /// TODO(opt): Profile and evaluate the usefulness of small buffer
  /// optimisation.
  llvm::SmallVector<Origin> AllOrigins;
  llvm::BumpPtrAllocator Allocator;
  llvm::DenseMap<const clang::ValueDecl *, OriginNode *> DeclToNode;
  llvm::DenseMap<const clang::Expr *, OriginNode *> ExprToNode;
  std::optional<OriginNode *> ThisOrigins;
  /// Types that are not inherently pointer-like but require origin tracking
  /// because of lifetime annotations (currently [[clang::lifetimebound]]) on
  /// functions that return them.
  llvm::DenseSet<const Type *> LifetimeAnnotatedOriginTypes;
  /// Fields accessed in the function body (or constructor init lists).
  /// Fields outside this set are excluded from origin tracking.
  llvm::SmallPtrSet<const FieldDecl *, 8> AccessedFields;

  /// Field-edge depth limit when building origin trees for record types:
  ///   - `std::nullopt`: no limit (full field tree).
  ///   - `0`: disable field tracking (records become single-origin).
  ///   - `N > 0`: track up to N levels of field edges.
  /// Pointee edges are not subject to this limit.
  std::optional<size_t> MaxFieldDepth = std::nullopt;
};
} // namespace clang::lifetimes::internal

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_ORIGINS_H
