//===-- lib/Semantics/check-omp-structure.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// OpenMP structure validity check list
//    1. invalid clauses on directive
//    2. invalid repeated clauses on directive
//    3. TODO: invalid nesting of regions

#ifndef FORTRAN_SEMANTICS_CHECK_OMP_STRUCTURE_H_
#define FORTRAN_SEMANTICS_CHECK_OMP_STRUCTURE_H_

#include "flang/Common/enum-set.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/semantics.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"

#include <unordered_map>

using OmpDirectiveSet = Fortran::common::EnumSet<llvm::omp::Directive,
    llvm::omp::Directive_enumSize>;

using OmpClauseSet =
    Fortran::common::EnumSet<llvm::omp::Clause, llvm::omp::Clause_enumSize>;

#define GEN_FLANG_DIRECTIVE_CLAUSE_SETS
#include "llvm/Frontend/OpenMP/OMP.cpp.inc"

namespace llvm {
namespace omp {
static OmpDirectiveSet parallelSet{Directive::OMPD_distribute_parallel_do,
    Directive::OMPD_distribute_parallel_do_simd, Directive::OMPD_parallel,
    Directive::OMPD_parallel_do, Directive::OMPD_parallel_do_simd,
    Directive::OMPD_parallel_sections, Directive::OMPD_parallel_workshare,
    Directive::OMPD_target_parallel, Directive::OMPD_target_parallel_do,
    Directive::OMPD_target_parallel_do_simd,
    Directive::OMPD_target_teams_distribute_parallel_do,
    Directive::OMPD_target_teams_distribute_parallel_do_simd,
    Directive::OMPD_teams_distribute_parallel_do,
    Directive::OMPD_teams_distribute_parallel_do_simd};
static OmpDirectiveSet doSet{Directive::OMPD_distribute_parallel_do,
    Directive::OMPD_distribute_parallel_do_simd, Directive::OMPD_parallel,
    Directive::OMPD_parallel_do, Directive::OMPD_parallel_do_simd,
    Directive::OMPD_do, Directive::OMPD_do_simd,
    Directive::OMPD_target_parallel_do, Directive::OMPD_target_parallel_do_simd,
    Directive::OMPD_target_teams_distribute_parallel_do,
    Directive::OMPD_target_teams_distribute_parallel_do_simd,
    Directive::OMPD_teams_distribute_parallel_do,
    Directive::OMPD_teams_distribute_parallel_do_simd};
static OmpDirectiveSet doSimdSet{Directive::OMPD_distribute_parallel_do_simd,
    Directive::OMPD_parallel_do_simd, Directive::OMPD_do_simd,
    Directive::OMPD_target_parallel_do_simd,
    Directive::OMPD_target_teams_distribute_parallel_do_simd,
    Directive::OMPD_teams_distribute_parallel_do_simd};
static OmpDirectiveSet taskloopSet{
    Directive::OMPD_taskloop, Directive::OMPD_taskloop_simd};
static OmpDirectiveSet targetSet{Directive::OMPD_target,
    Directive::OMPD_target_parallel, Directive::OMPD_target_parallel_do,
    Directive::OMPD_target_parallel_do_simd, Directive::OMPD_target_simd,
    Directive::OMPD_target_teams, Directive::OMPD_target_teams_distribute,
    Directive::OMPD_target_teams_distribute_simd};
static OmpDirectiveSet simdSet{Directive::OMPD_distribute_parallel_do_simd,
    Directive::OMPD_distribute_simd, Directive::OMPD_parallel_do_simd,
    Directive::OMPD_do_simd, Directive::OMPD_simd,
    Directive::OMPD_target_parallel_do_simd,
    Directive::OMPD_target_teams_distribute_parallel_do_simd,
    Directive::OMPD_target_teams_distribute_simd, Directive::OMPD_target_simd,
    Directive::OMPD_taskloop_simd,
    Directive::OMPD_teams_distribute_parallel_do_simd,
    Directive::OMPD_teams_distribute_simd};
static OmpDirectiveSet taskGeneratingSet{
    OmpDirectiveSet{Directive::OMPD_task} | taskloopSet};
} // namespace omp
} // namespace llvm

namespace Fortran::semantics {

class OmpStructureChecker : public virtual BaseChecker {
public:
  OmpStructureChecker(SemanticsContext &context) : context_{context} {}

  void Enter(const parser::OpenMPConstruct &);
  void Enter(const parser::OpenMPLoopConstruct &);
  void Leave(const parser::OpenMPLoopConstruct &);
  void Enter(const parser::OmpEndLoopDirective &);

  void Enter(const parser::OpenMPBlockConstruct &);
  void Leave(const parser::OpenMPBlockConstruct &);
  void Enter(const parser::OmpEndBlockDirective &);

  void Enter(const parser::OpenMPSectionsConstruct &);
  void Leave(const parser::OpenMPSectionsConstruct &);
  void Enter(const parser::OmpEndSectionsDirective &);

  void Enter(const parser::OpenMPDeclareSimdConstruct &);
  void Leave(const parser::OpenMPDeclareSimdConstruct &);
  void Enter(const parser::OpenMPDeclareTargetConstruct &);
  void Leave(const parser::OpenMPDeclareTargetConstruct &);

  void Enter(const parser::OpenMPSimpleStandaloneConstruct &);
  void Leave(const parser::OpenMPSimpleStandaloneConstruct &);
  void Enter(const parser::OpenMPFlushConstruct &);
  void Leave(const parser::OpenMPFlushConstruct &);
  void Enter(const parser::OpenMPCancelConstruct &);
  void Leave(const parser::OpenMPCancelConstruct &);
  void Enter(const parser::OpenMPCancellationPointConstruct &);
  void Leave(const parser::OpenMPCancellationPointConstruct &);

  void Leave(const parser::OmpClauseList &);
  void Enter(const parser::OmpClause &);
  void Enter(const parser::OmpNowait &);
  void Enter(const parser::OmpClause::Inbranch &);
  void Enter(const parser::OmpClause::Mergeable &);
  void Enter(const parser::OmpClause::Nogroup &);
  void Enter(const parser::OmpClause::Notinbranch &);
  void Enter(const parser::OmpClause::Untied &);
  void Enter(const parser::OmpClause::Collapse &);
  void Enter(const parser::OmpClause::Copyin &);
  void Enter(const parser::OmpClause::Copyprivate &);
  void Enter(const parser::OmpClause::Device &);
  void Enter(const parser::OmpClause::DistSchedule &);
  void Enter(const parser::OmpClause::Final &);
  void Enter(const parser::OmpClause::Firstprivate &);
  void Enter(const parser::OmpClause::From &);
  void Enter(const parser::OmpClause::Grainsize &);
  void Enter(const parser::OmpClause::Lastprivate &);
  void Enter(const parser::OmpClause::NumTasks &);
  void Enter(const parser::OmpClause::NumTeams &);
  void Enter(const parser::OmpClause::NumThreads &);
  void Enter(const parser::OmpClause::Ordered &);
  void Enter(const parser::OmpClause::Priority &);
  void Enter(const parser::OmpClause::Private &);
  void Enter(const parser::OmpClause::Safelen &);
  void Enter(const parser::OmpClause::Shared &);
  void Enter(const parser::OmpClause::Simdlen &);
  void Enter(const parser::OmpClause::ThreadLimit &);
  void Enter(const parser::OmpClause::To &);
  void Enter(const parser::OmpClause::Link &);
  void Enter(const parser::OmpClause::Uniform &);
  void Enter(const parser::OmpClause::UseDevicePtr &);
  void Enter(const parser::OmpClause::IsDevicePtr &);

  void Enter(const parser::OmpAlignedClause &);
  void Enter(const parser::OmpDefaultClause &);
  void Enter(const parser::OmpDefaultmapClause &);
  void Enter(const parser::OmpDependClause &);
  void Enter(const parser::OmpIfClause &);
  void Enter(const parser::OmpLinearClause &);
  void Enter(const parser::OmpMapClause &);
  void Enter(const parser::OmpProcBindClause &);
  void Enter(const parser::OmpReductionClause &);
  void Enter(const parser::OmpScheduleClause &);

private:
#define GEN_FLANG_DIRECTIVE_CLAUSE_MAP
#include "llvm/Frontend/OpenMP/OMP.cpp.inc"

  struct OmpContext {
    OmpContext(parser::CharBlock source, llvm::omp::Directive d)
        : directiveSource{source}, directive{d} {}
    parser::CharBlock directiveSource{nullptr};
    parser::CharBlock clauseSource{nullptr};
    llvm::omp::Directive directive;
    OmpClauseSet allowedClauses{};
    OmpClauseSet allowedOnceClauses{};
    OmpClauseSet allowedExclusiveClauses{};
    OmpClauseSet requiredClauses{};

    const parser::OmpClause *clause{nullptr};
    std::multimap<llvm::omp::Clause, const parser::OmpClause *> clauseInfo;
  };
  // back() is the top of the stack
  OmpContext &GetContext() {
    CHECK(!ompContext_.empty());
    return ompContext_.back();
  }
  // reset source location, check information, and
  // collected information for END directive
  void ResetPartialContext(const parser::CharBlock &source) {
    CHECK(!ompContext_.empty());
    SetContextDirectiveSource(source);
    GetContext().allowedClauses = {};
    GetContext().allowedOnceClauses = {};
    GetContext().allowedExclusiveClauses = {};
    GetContext().requiredClauses = {};
    GetContext().clauseInfo = {};
  }
  void SetContextDirectiveSource(const parser::CharBlock &directive) {
    GetContext().directiveSource = directive;
  }
  void SetContextClause(const parser::OmpClause &clause) {
    GetContext().clauseSource = clause.source;
    GetContext().clause = &clause;
  }
  void SetContextDirectiveEnum(llvm::omp::Directive dir) {
    GetContext().directive = dir;
  }
  void SetContextAllowed(const OmpClauseSet &allowed) {
    GetContext().allowedClauses = allowed;
  }
  void SetContextAllowedOnce(const OmpClauseSet &allowedOnce) {
    GetContext().allowedOnceClauses = allowedOnce;
  }
  void SetContextAllowedExclusive(const OmpClauseSet &allowedExclusive) {
    GetContext().allowedExclusiveClauses = allowedExclusive;
  }
  void SetContextRequired(const OmpClauseSet &required) {
    GetContext().requiredClauses = required;
  }
  void SetContextClauseInfo(llvm::omp::Clause type) {
    GetContext().clauseInfo.emplace(type, GetContext().clause);
  }
  const parser::OmpClause *FindClause(llvm::omp::Clause type) {
    auto it{GetContext().clauseInfo.find(type)};
    if (it != GetContext().clauseInfo.end()) {
      return it->second;
    }
    return nullptr;
  }
  void PushContext(const parser::CharBlock &source, llvm::omp::Directive dir) {
    ompContext_.emplace_back(source, dir);
  }
  void SetClauseSets(llvm::omp::Directive dir) {
    ompContext_.back().allowedClauses = directiveClausesTable[dir].allowed;
    ompContext_.back().allowedOnceClauses =
        directiveClausesTable[dir].allowedOnce;
    ompContext_.back().allowedExclusiveClauses =
        directiveClausesTable[dir].allowedExclusive;
    ompContext_.back().requiredClauses =
        directiveClausesTable[dir].requiredOneOf;
  }
  void PushContextAndClauseSets(
      const parser::CharBlock &source, llvm::omp::Directive dir) {
    PushContext(source, dir);
    SetClauseSets(dir);
  }
  void RequiresConstantPositiveParameter(
      const llvm::omp::Clause &clause, const parser::ScalarIntConstantExpr &i);
  void RequiresPositiveParameter(
      const llvm::omp::Clause &clause, const parser::ScalarIntExpr &i);

  bool CurrentDirectiveIsNested() { return ompContext_.size() > 0; };
  bool HasInvalidWorksharingNesting(
      const parser::CharBlock &, const OmpDirectiveSet &);
  void CheckAllowed(llvm::omp::Clause);
  void CheckRequired(llvm::omp::Clause);
  std::string ContextDirectiveAsFortran();
  void SayNotMatching(const parser::CharBlock &, const parser::CharBlock &);
  template <typename A, typename B, typename C>
  const A &CheckMatching(const B &beginDir, const C &endDir) {
    const A &begin{std::get<A>(beginDir.t)};
    const A &end{std::get<A>(endDir.t)};
    if (begin.v != end.v) {
      SayNotMatching(begin.source, end.source);
    }
    return begin;
  }

  // specific clause related
  bool ScheduleModifierHasType(const parser::OmpScheduleClause &,
      const parser::OmpScheduleModifierType::ModType &);

  SemanticsContext &context_;
  std::vector<OmpContext> ompContext_; // used as a stack
};
} // namespace Fortran::semantics
#endif // FORTRAN_SEMANTICS_CHECK_OMP_STRUCTURE_H_
