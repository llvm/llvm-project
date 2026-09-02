//===--- CIRGenOpenMPConstructDecomposition.h -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENOPENMPCONSTRUCTDECOMPOSITION_H
#define LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENOPENMPCONSTRUCTDECOMPOSITION_H

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/OpenMPClause.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/Basic/OpenMPKinds.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Frontend/OpenMP/ClauseT.h"
#include "llvm/Frontend/OpenMP/ConstructDecompositionT.h"
#include "llvm/Frontend/OpenMP/OMP.h"
#include "llvm/Support/Casting.h"

#include <cassert>
#include <optional>
#include <utility>

namespace clang::CIRGen::omp {

// tomp type parameters for Clang: objects are identified by their canonical
// declaration, expressions and types are the corresponding AST nodes.
using TypeTy = const clang::Type *;
using IdTy = const clang::ValueDecl *;
using ExprTy = const clang::Expr *;

} // namespace clang::CIRGen::omp

// The decomposition operates on tomp::ObjectT<IdTy, ExprTy>; provide the
// specialization for our identity/expression types.
namespace tomp::type {
template <>
struct ObjectT<clang::CIRGen::omp::IdTy, clang::CIRGen::omp::ExprTy> {
  using IdTy = clang::CIRGen::omp::IdTy;
  using ExprTy = clang::CIRGen::omp::ExprTy;

  IdTy id() const { return identity; }
  const std::optional<ExprTy> &ref() const { return designator; }

  bool operator<(const ObjectT &other) const {
    return identity < other.identity;
  }

  IdTy identity = nullptr;
  std::optional<ExprTy> designator;
};
} // namespace tomp::type

namespace clang::CIRGen::omp {

using Object = tomp::ObjectT<IdTy, ExprTy>;
using ObjectList = tomp::ObjectListT<IdTy, ExprTy>;
using ClauseBase = tomp::ClauseT<TypeTy, IdTy, ExprTy>;

/// A tomp clause that remembers the Clang AST clause it came from, so the
/// existing emitters can emit it after the decomposition assigns it to a leaf.
/// Synthesized clauses have no AST counterpart and leave `original` null.
struct Clause : public ClauseBase {
  Clause() = default;
  Clause(ClauseBase &&base) : ClauseBase(std::move(base)) {}
  const clang::OMPClause *original = nullptr;
};

inline const clang::ValueDecl *getBaseValueDecl(const clang::Expr *e) {
  e = e->IgnoreParenImpCasts();
  for (;;) {
    if (const auto *ase = llvm::dyn_cast<clang::ArraySubscriptExpr>(e)) {
      e = ase->getBase()->IgnoreParenImpCasts();
      continue;
    }
    if (const auto *ase = llvm::dyn_cast<clang::ArraySectionExpr>(e)) {
      e = ase->getBase()->IgnoreParenImpCasts();
      continue;
    }
    break;
  }
  if (const auto *dre = llvm::dyn_cast<clang::DeclRefExpr>(e))
    return llvm::cast<clang::ValueDecl>(dre->getDecl()->getCanonicalDecl());
  if (const auto *me = llvm::dyn_cast<clang::MemberExpr>(e))
    return llvm::cast<clang::ValueDecl>(
        me->getMemberDecl()->getCanonicalDecl());
  return nullptr;
}

/// Build a tomp Object whose identity is the base variable's canonical decl.
inline Object makeObject(const clang::Expr *e) {
  return Object{getBaseValueDecl(e), e};
}

/// Build the tomp object list from a Clang var-list clause.
template <typename ClangClause>
inline ObjectList makeObjects(const ClangClause &c) {
  ObjectList list;
  for (const clang::Expr *e : c.getVarRefs())
    list.push_back(makeObject(e));
  return list;
}

/// Wrap a tomp clause payload into a Clause, remembering its AST origin.
template <typename Specific>
inline Clause makeClause(llvm::omp::Clause id, Specific &&specific,
                         const clang::OMPClause &original) {
  Clause c{ClauseBase{id, std::forward<Specific>(specific)}};
  c.original = &original;
  return c;
}

/// Clause kinds that need a dedicated conversion: they either have a specific
/// applyClause() overload (so the payload type selects it) or their contents
/// feed the algorithm. Guards the generic fallback in makeGeneric.
inline bool needsSpecificHandling(llvm::omp::Clause kind) {
  switch (kind) {
  case llvm::omp::OMPC_allocate:
  case llvm::omp::OMPC_collapse:
  case llvm::omp::OMPC_default:
  case llvm::omp::OMPC_dyn_groupprivate:
  case llvm::omp::OMPC_firstprivate:
  case llvm::omp::OMPC_if:
  case llvm::omp::OMPC_lastprivate:
  case llvm::omp::OMPC_linear:
  case llvm::omp::OMPC_map:
  case llvm::omp::OMPC_nowait:
  case llvm::omp::OMPC_ompx_attribute:
  case llvm::omp::OMPC_ompx_bare:
  case llvm::omp::OMPC_order:
  case llvm::omp::OMPC_private:
  case llvm::omp::OMPC_reduction:
  case llvm::omp::OMPC_shared:
  case llvm::omp::OMPC_thread_limit:
    return true;
  default:
    return false;
  }
}

/// Clause kinds CIR is able to emit today.
inline bool isEmittableClause(llvm::omp::Clause kind) {
  switch (kind) {
  case llvm::omp::OMPC_map:
  case llvm::omp::OMPC_proc_bind:
    return true;
  default:
    return false;
  }
}

/// Represent a clause by kind only, using an inert empty payload that routes
/// through the algorithm's generic applyClause() path (which reads just the
/// clause id). Valid for any clause with no specific applyClause() overload.
inline Clause makeGeneric(llvm::omp::Clause id, const clang::OMPClause &orig) {
  assert((!isEmittableClause(id) || !needsSpecificHandling(id)) &&
         "CIR-emittable clause needs specific decomposition handling");
  return makeClause(id, tomp::clause::ThreadsT<TypeTy, IdTy, ExprTy>{}, orig);
}

/// Convert a single Clang clause to its tomp representation. Every kind is
/// handled; contents are populated only where the algorithm reads them.
inline Clause convertClause(const clang::OMPClause &c) {
  namespace tc = tomp::clause;
  const llvm::omp::Clause kind = c.getClauseKind();
  switch (kind) {
  // Clauses whose contents the algorithm inspects.
  case llvm::omp::OMPC_map: {
    tc::MapT<TypeTy, IdTy, ExprTy> m{
        {/*MapType=*/std::nullopt, /*MapTypeModifiers=*/std::nullopt,
         /*AttachModifier=*/std::nullopt, /*RefModifier=*/std::nullopt,
         /*Mappers=*/std::nullopt, /*Iterator=*/std::nullopt,
         /*LocatorList=*/makeObjects(llvm::cast<clang::OMPMapClause>(c))}};
    return makeClause(kind, std::move(m), c);
  }
  case llvm::omp::OMPC_firstprivate:
    return makeClause(
        kind,
        tc::FirstprivateT<TypeTy, IdTy, ExprTy>{
            /*List=*/makeObjects(llvm::cast<clang::OMPFirstprivateClause>(c))},
        c);
  case llvm::omp::OMPC_private:
    return makeClause(kind,
                      tc::PrivateT<TypeTy, IdTy, ExprTy>{/*List=*/makeObjects(
                          llvm::cast<clang::OMPPrivateClause>(c))},
                      c);
  case llvm::omp::OMPC_shared:
    return makeClause(kind,
                      tc::SharedT<TypeTy, IdTy, ExprTy>{/*List=*/makeObjects(
                          llvm::cast<clang::OMPSharedClause>(c))},
                      c);
  case llvm::omp::OMPC_lastprivate:
    return makeClause(
        kind,
        tc::LastprivateT<TypeTy, IdTy, ExprTy>{
            {/*LastprivateModifier=*/std::nullopt,
             /*List=*/makeObjects(llvm::cast<clang::OMPLastprivateClause>(c))}},
        c);
  case llvm::omp::OMPC_linear:
    return makeClause(
        kind,
        tc::LinearT<TypeTy, IdTy, ExprTy>{
            {/*StepComplexModifier=*/std::nullopt,
             /*LinearModifier=*/std::nullopt,
             /*List=*/makeObjects(llvm::cast<clang::OMPLinearClause>(c))}},
        c);
  case llvm::omp::OMPC_reduction:
    return makeClause(
        kind,
        tc::ReductionT<TypeTy, IdTy, ExprTy>{
            {/*ReductionModifier=*/std::nullopt, /*ReductionIdentifiers=*/{},
             /*List=*/makeObjects(llvm::cast<clang::OMPReductionClause>(c))}},
        c);
  case llvm::omp::OMPC_if: {
    const auto &ic = llvm::cast<clang::OMPIfClause>(c);
    std::optional<llvm::omp::Directive> mod;
    if (ic.getNameModifier() != llvm::omp::OMPD_unknown)
      mod = ic.getNameModifier();
    return makeClause(
        kind,
        tc::IfT<TypeTy, IdTy, ExprTy>{{/*DirectiveNameModifier=*/mod,
                                       /*IfExpression=*/ic.getCondition()}},
        c);
  }
  // Clauses with a specific applyClause() overload but no contents the
  // algorithm reads: carry the correct payload type so dispatch selects it.
  case llvm::omp::OMPC_allocate:
    return makeClause(kind,
                      tc::AllocateT<TypeTy, IdTy, ExprTy>{
                          {std::nullopt, std::nullopt, /*List=*/{}}},
                      c);
  case llvm::omp::OMPC_collapse:
    return makeClause(kind, tc::CollapseT<TypeTy, IdTy, ExprTy>{/*N=*/nullptr},
                      c);
  case llvm::omp::OMPC_default:
    return makeClause(
        kind,
        tc::DefaultT<TypeTy, IdTy, ExprTy>{
            tc::DefaultT<TypeTy, IdTy, ExprTy>::DataSharingAttribute::Shared},
        c);
  case llvm::omp::OMPC_dyn_groupprivate:
    return makeClause(kind,
                      tc::DynGroupprivateT<TypeTy, IdTy, ExprTy>{
                          {std::nullopt, std::nullopt, /*Size=*/nullptr}},
                      c);
  case llvm::omp::OMPC_nowait:
    return makeClause(kind, tc::NowaitT<TypeTy, IdTy, ExprTy>{}, c);
  case llvm::omp::OMPC_ompx_attribute:
    return makeClause(kind, tc::OmpxAttributeT<TypeTy, IdTy, ExprTy>{}, c);
  case llvm::omp::OMPC_ompx_bare:
    return makeClause(kind, tc::OmpxBareT<TypeTy, IdTy, ExprTy>{}, c);
  case llvm::omp::OMPC_order:
    return makeClause(
        kind,
        tc::OrderT<TypeTy, IdTy, ExprTy>{
            {std::nullopt,
             tc::OrderT<TypeTy, IdTy, ExprTy>::Ordering::Concurrent}},
        c);
  case llvm::omp::OMPC_thread_limit:
    return makeClause(kind, tc::ThreadLimitT<TypeTy, IdTy, ExprTy>{/*List=*/{}},
                      c);
  // Everything else routes by kind alone.
  default:
    return makeGeneric(kind, c);
  }
}

/// Helper required by ConstructDecompositionT.
struct DecompositionHelper {
  /// Our object identities are already normalized to the base variable's decl,
  /// so an object is its own base.
  std::optional<Object> getBaseObject(const Object &object) const {
    return object;
  }
  /// CIR does not lower loop directives yet, so there is no iteration variable.
  std::optional<Object> getLoopIterVar() const { return std::nullopt; }
};

struct LeafWithClauses {
  llvm::omp::Directive id = llvm::omp::Directive::OMPD_unknown;
  llvm::SmallVector<const clang::OMPClause *> clauses;
  llvm::SmallVector<llvm::omp::Clause> synthesized;
};

using ConstructQueue = llvm::SmallVector<LeafWithClauses>;

/// Given a potentially compound directive with a list of clauses that apply to
/// it, break it up into individual leaf constructs each with the subset of
/// applicable clauses (plus implicit clauses, if any). From that create a work
/// queue, ordered outermost to innermost, where each work item corresponds to
/// the leaf construct with its clauses. Implicit clauses are synthesized by the
/// decomposition and have no Clang AST node, so they are listed separately.
inline ConstructQueue buildConstructQueue(unsigned openmpVersion,
                                          const OMPExecutableDirective &s) {
  llvm::SmallVector<Clause> input;
  for (const OMPClause *c : s.clauses())
    input.push_back(convertClause(*c));

  DecompositionHelper helper;
  tomp::ConstructDecompositionT<Clause, DecompositionHelper> decomp(
      llvm::omp::Version(openmpVersion), helper, s.getDirectiveKind(),
      llvm::ArrayRef<Clause>(input));

  ConstructQueue result;
  for (const tomp::DirectiveWithClauses<Clause> &dwc : decomp.output) {
    LeafWithClauses leaf;
    leaf.id = dwc.id;
    for (const Clause &c : dwc.clauses) {
      if (c.original)
        leaf.clauses.push_back(c.original);
      else
        leaf.synthesized.push_back(c.id);
    }
    result.push_back(std::move(leaf));
  }
  return result;
}

inline bool isLastItemInQueue(ConstructQueue::const_iterator item,
                              const ConstructQueue &queue) {
  return std::next(item) == queue.end();
}

} // namespace clang::CIRGen::omp

#endif // LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENOPENMPCONSTRUCTDECOMPOSITION_H
