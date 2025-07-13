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

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_FINDTARGET_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_FINDTARGET_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <bitset>

namespace clang {

class HeuristicResolver;

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
/// Note that this function only returns NamedDecls. Generally other decls
/// don't have references in this sense, just the node itself.
/// If callers want to support such decls, they should cast the node directly.
///
/// FIXME: some AST nodes cannot be DynTypedNodes, these cannot be specified.
llvm::SmallVector<const NamedDecl *, 1>
targetDecl(const DynTypedNode &, DeclRelationSet Mask,
           const HeuristicResolver *Resolver);

/// Similar to targetDecl(), however instead of applying a filter, all possible
/// decls are returned along with their DeclRelationSets.
/// This is suitable for indexing, where everything is recorded and filtering
/// is applied later.
llvm::SmallVector<std::pair<const NamedDecl *, DeclRelationSet>, 1>
allTargetDecls(const DynTypedNode &, const HeuristicResolver *);

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
  // e.g. namespace client { [[X]] x; }

  /// This declaration is an alias that was referred to.
  /// e.g. using ns::X (the UsingDecl directly referenced),
  ///      using Z = ns::Y (the TypeAliasDecl directly referenced)
  Alias,
  /// This is the underlying declaration for a renaming-alias, decltype etc.
  /// e.g. class ns::Y (the underlying declaration referenced).
  ///
  /// Note that we don't treat `using ns::X` as a first-class declaration like
  /// `using Z = ns::Y`. Therefore reference to X that goes through this
  /// using-decl is considered a direct reference (without the Underlying bit).
  /// Nevertheless, we report `using ns::X` as an Alias, so that some features
  /// like go-to-definition can still target it.
  Underlying,
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &, DeclRelation);

/// Information about a reference written in the source code, independent of the
/// actual AST node that this reference lives in.
/// Useful for tools that are source-aware, e.g. refactorings.
struct ReferenceLoc {
  /// Contains qualifier written in the code, if any, e.g. 'ns::' for 'ns::foo'.
  NestedNameSpecifierLoc Qualifier;
  /// Start location of the last name part, i.e. 'foo' in 'ns::foo<int>'.
  SourceLocation NameLoc;
  /// True if the reference is a declaration or definition;
  bool IsDecl = false;
  // FIXME: add info about template arguments.
  /// A list of targets referenced by this name. Normally this has a single
  /// element, but multiple is also possible, e.g. in case of using declarations
  /// or unresolved overloaded functions.
  /// For dependent and unresolved references, Targets can also be empty.
  llvm::SmallVector<const NamedDecl *, 1> Targets;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, ReferenceLoc R);

/// Recursively traverse \p S and report all references explicitly written in
/// the code. The main use-case is refactorings that need to process all
/// references in some subrange of the file and apply simple edits, e.g. add
/// qualifiers.
/// FIXME: currently this does not report references to overloaded operators.
/// FIXME: extend to report location information about declaration names too.
void findExplicitReferences(const Stmt *S,
                            llvm::function_ref<void(ReferenceLoc)> Out,
                            const HeuristicResolver *Resolver);
void findExplicitReferences(const Decl *D,
                            llvm::function_ref<void(ReferenceLoc)> Out,
                            const HeuristicResolver *Resolver);
void findExplicitReferences(const ASTContext &AST,
                            llvm::function_ref<void(ReferenceLoc)> Out,
                            const HeuristicResolver *Resolver);

/// Find declarations explicitly referenced in the source code defined by \p N.
/// For templates, will prefer to return a template instantiation whenever
/// possible. However, can also return a template pattern if the specialization
/// cannot be picked, e.g. in dependent code or when there is no corresponding
/// Decl for a template instantiation, e.g. for templated using decls:
///    template <class T> using Ptr = T*;
///    Ptr<int> x;
///    ^~~ there is no Decl for 'Ptr<int>', so we return the template pattern.
/// \p Mask should not contain TemplatePattern or TemplateInstantiation.
llvm::SmallVector<const NamedDecl *, 1>
explicitReferenceTargets(DynTypedNode N, DeclRelationSet Mask,
                         const HeuristicResolver *Resolver);

// Boring implementation details of bitfield.

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
  bool contains(DeclRelationSet Other) const {
    return (S & Other.S) == Other.S;
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

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_FINDTARGET_H
