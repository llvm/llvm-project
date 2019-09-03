//===--- FindTarget.h - What does an AST node refer to? ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Many clangd features are concerned with references in the AST:
//  - xrefs, go-to-definition, explicitly talk about references
//  - hover and code actions relate to things you "target" in the editor
//  - refactoring actions need to know about entities that are referenced
//    to determine whether/how the edit can be applied.
//
// Historically, we have used libIndex (IndexDataConsumer) to tie source
// locations to referenced declarations. This file defines a more decoupled
// approach based around AST nodes (DynTypedNode), and can be combined with
// SelectionTree or other traversals.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTTypeTraits.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <bitset>

namespace clang {
namespace clangd {
/// Describes the link between an AST node and a Decl it refers to.
enum class DeclRelation : unsigned;
/// A bitfield of DeclRelations.
class DeclRelationSet;

/// targetDecl() finds the declaration referred to by an AST node.
/// For example a RecordTypeLoc refers to the RecordDecl for the type.
///
/// In some cases there are multiple results, e.g. a dependent unresolved
/// OverloadExpr may have several candidates. All will be returned:
///
///    void foo(int);    <-- candidate
///    void foo(double); <-- candidate
///    template <typename T> callFoo() { foo(T()); }
///                                      ^ OverloadExpr
///
/// In other cases, there may be choices about what "referred to" means.
/// e.g. does naming a typedef refer to the underlying type?
/// The results are marked with a set of DeclRelations, and can be filtered.
///
///    struct S{};    <-- candidate (underlying)
///    using T = S{}; <-- candidate (alias)
///    T x;
///    ^ TypedefTypeLoc
///
/// Formally, we walk a graph starting at the provided node, and return the
/// decls that were found. Certain edges in the graph have labels, and for each
/// decl we return the set of labels seen on a path to the decl.
/// For the previous example:
///
///                TypedefTypeLoc T
///                       |
///                 TypedefType T
///                    /     \
///           [underlying]  [alias]
///                  /         \
///          RecordDecl S    TypeAliasDecl T
///
/// FIXME: some AST nodes cannot be DynTypedNodes, these cannot be specified.
llvm::SmallVector<const Decl *, 1>
targetDecl(const ast_type_traits::DynTypedNode &, DeclRelationSet Mask);

/// Similar to targetDecl(), however instead of applying a filter, all possible
/// decls are returned along with their DeclRelationSets.
/// This is suitable for indexing, where everything is recorded and filtering
/// is applied later.
llvm::SmallVector<std::pair<const Decl *, DeclRelationSet>, 1>
allTargetDecls(const ast_type_traits::DynTypedNode &);

enum class DeclRelation : unsigned {
  // Template options apply when the declaration is an instantiated template.
  // e.g. [[vector<int>]] vec;

  /// This is the template instantiation that was referred to.
  /// e.g. template<> class vector<int> (the implicit specialization)
  TemplateInstantiation,
  /// This is the pattern the template specialization was instantiated from.
  /// e.g. class vector<T> (the pattern within the primary template)
  TemplatePattern,

  // Alias options apply when the declaration is an alias.
  // e.g. namespace clang { [[StringRef]] S; }

  /// This declaration is an alias that was referred to.
  /// e.g. using llvm::StringRef (the UsingDecl directly referenced).
  Alias,
  /// This is the underlying declaration for an alias, decltype etc.
  /// e.g. class llvm::StringRef (the underlying declaration referenced).
  Underlying,
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &, DeclRelation);

class DeclRelationSet {
  using Set = std::bitset<static_cast<unsigned>(DeclRelation::Underlying) + 1>;
  Set S;
  DeclRelationSet(Set S) : S(S) {}

public:
  DeclRelationSet() = default;
  DeclRelationSet(DeclRelation R) { S.set(static_cast<unsigned>(R)); }

  explicit operator bool() const { return S.any(); }
  friend DeclRelationSet operator&(DeclRelationSet L, DeclRelationSet R) {
    return L.S & R.S;
  }
  friend DeclRelationSet operator|(DeclRelationSet L, DeclRelationSet R) {
    return L.S | R.S;
  }
  friend bool operator==(DeclRelationSet L, DeclRelationSet R) {
    return L.S == R.S;
  }
  friend DeclRelationSet operator~(DeclRelationSet R) { return ~R.S; }
  DeclRelationSet &operator|=(DeclRelationSet Other) {
    S |= Other.S;
    return *this;
  }
  DeclRelationSet &operator&=(DeclRelationSet Other) {
    S &= Other.S;
    return *this;
  }
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &, DeclRelationSet);
};
// The above operators can't be looked up if both sides are enums.
// over.match.oper.html#3.2
inline DeclRelationSet operator|(DeclRelation L, DeclRelation R) {
  return DeclRelationSet(L) | DeclRelationSet(R);
}
inline DeclRelationSet operator&(DeclRelation L, DeclRelation R) {
  return DeclRelationSet(L) & DeclRelationSet(R);
}
inline DeclRelationSet operator~(DeclRelation R) { return ~DeclRelationSet(R); }
llvm::raw_ostream &operator<<(llvm::raw_ostream &, DeclRelationSet);

} // namespace clangd
} // namespace clang
