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

namespace Fortran::semantics {

ENUM_CLASS(OmpDirective, ATOMIC, BARRIER, CANCEL, CANCELLATION_POINT, CRITICAL,
    DECLARE_REDUCTION, DECLARE_SIMD, DECLARE_TARGET, DISTRIBUTE,
    DISTRIBUTE_PARALLEL_DO, DISTRIBUTE_PARALLEL_DO_SIMD, DISTRIBUTE_SIMD, DO,
    DO_SIMD, END_CRITICAL, END_DO, END_DO_SIMD, END_SECTIONS, END_SINGLE,
    END_WORKSHARE, FLUSH, MASTER, ORDERED, PARALLEL, PARALLEL_DO,
    PARALLEL_DO_SIMD, PARALLEL_SECTIONS, PARALLEL_WORKSHARE, SECTION, SECTIONS,
    SIMD, SINGLE, TARGET, TARGET_DATA, TARGET_ENTER_DATA, TARGET_EXIT_DATA,
    TARGET_PARALLEL, TARGET_PARALLEL_DO, TARGET_PARALLEL_DO_SIMD, TARGET_SIMD,
    TARGET_TEAMS, TARGET_TEAMS_DISTRIBUTE, TARGET_TEAMS_DISTRIBUTE_PARALLEL_DO,
    TARGET_TEAMS_DISTRIBUTE_PARALLEL_DO_SIMD, TARGET_TEAMS_DISTRIBUTE_SIMD,
    TARGET_UPDATE, TASK, TASKGROUP, TASKLOOP, TASKLOOP_SIMD, TASKWAIT,
    TASKYIELD, TEAMS, TEAMS_DISTRIBUTE, TEAMS_DISTRIBUTE_PARALLEL_DO,
    TEAMS_DISTRIBUTE_PARALLEL_DO_SIMD, TEAMS_DISTRIBUTE_SIMD, THREADPRIVATE,
    WORKSHARE)

using OmpDirectiveSet = common::EnumSet<OmpDirective, OmpDirective_enumSize>;

ENUM_CLASS(OmpClause, ALIGNED, COLLAPSE, COPYIN, COPYPRIVATE, DEFAULT,
    DEFAULTMAP, DEPEND, DEVICE, DIST_SCHEDULE, FINAL, FIRSTPRIVATE, FROM,
    GRAINSIZE, IF, INBRANCH, IS_DEVICE_PTR, LASTPRIVATE, LINEAR, LINK, MAP,
    MERGEABLE, NOGROUP, NOTINBRANCH, NOWAIT, NUM_TASKS, NUM_TEAMS, NUM_THREADS,
    ORDERED, PRIORITY, PRIVATE, PROC_BIND, REDUCTION, SAFELEN, SCHEDULE, SHARED,
    SIMD, SIMDLEN, THREAD_LIMIT, THREADS, TO, UNIFORM, UNTIED, USE_DEVICE_PTR)

using OmpClauseSet = common::EnumSet<OmpClause, OmpClause_enumSize>;

static constexpr OmpDirectiveSet parallelSet{
    OmpDirective::DISTRIBUTE_PARALLEL_DO,
    OmpDirective::DISTRIBUTE_PARALLEL_DO_SIMD, OmpDirective::PARALLEL,
    OmpDirective::PARALLEL_DO, OmpDirective::PARALLEL_DO_SIMD,
    OmpDirective::PARALLEL_SECTIONS, OmpDirective::PARALLEL_WORKSHARE,
    OmpDirective::TARGET_PARALLEL, OmpDirective::TARGET_PARALLEL_DO,
    OmpDirective::TARGET_PARALLEL_DO_SIMD,
    OmpDirective::TARGET_TEAMS_DISTRIBUTE_PARALLEL_DO,
    OmpDirective::TARGET_TEAMS_DISTRIBUTE_PARALLEL_DO_SIMD,
    OmpDirective::TEAMS_DISTRIBUTE_PARALLEL_DO,
    OmpDirective::TEAMS_DISTRIBUTE_PARALLEL_DO_SIMD};
static constexpr OmpDirectiveSet doSet{OmpDirective::DISTRIBUTE_PARALLEL_DO,
    OmpDirective::DISTRIBUTE_PARALLEL_DO_SIMD, OmpDirective::PARALLEL_DO,
    OmpDirective::PARALLEL_DO_SIMD, OmpDirective::DO, OmpDirective::DO_SIMD,
    OmpDirective::TARGET_PARALLEL_DO, OmpDirective::TARGET_PARALLEL_DO_SIMD,
    OmpDirective::TARGET_TEAMS_DISTRIBUTE_PARALLEL_DO,
    OmpDirective::TARGET_TEAMS_DISTRIBUTE_PARALLEL_DO_SIMD,
    OmpDirective::TEAMS_DISTRIBUTE_PARALLEL_DO,
    OmpDirective::TEAMS_DISTRIBUTE_PARALLEL_DO_SIMD};
static constexpr OmpDirectiveSet doSimdSet{
    OmpDirective::DISTRIBUTE_PARALLEL_DO_SIMD, OmpDirective::PARALLEL_DO_SIMD,
    OmpDirective::DO_SIMD, OmpDirective::TARGET_PARALLEL_DO_SIMD,
    OmpDirective::TARGET_TEAMS_DISTRIBUTE_PARALLEL_DO_SIMD,
    OmpDirective::TEAMS_DISTRIBUTE_PARALLEL_DO_SIMD};
static constexpr OmpDirectiveSet taskloopSet{
    OmpDirective::TASKLOOP, OmpDirective::TASKLOOP_SIMD};
static constexpr OmpDirectiveSet targetSet{OmpDirective::TARGET,
    OmpDirective::TARGET_PARALLEL, OmpDirective::TARGET_PARALLEL_DO,
    OmpDirective::TARGET_PARALLEL_DO_SIMD, OmpDirective::TARGET_SIMD,
    OmpDirective::TARGET_TEAMS, OmpDirective::TARGET_TEAMS_DISTRIBUTE,
    OmpDirective::TARGET_TEAMS_DISTRIBUTE_PARALLEL_DO,
    OmpDirective::TARGET_TEAMS_DISTRIBUTE_PARALLEL_DO_SIMD,
    OmpDirective::TARGET_TEAMS_DISTRIBUTE_SIMD};
static constexpr OmpDirectiveSet simdSet{
    OmpDirective::DISTRIBUTE_PARALLEL_DO_SIMD, OmpDirective::DISTRIBUTE_SIMD,
    OmpDirective::PARALLEL_DO_SIMD, OmpDirective::DO_SIMD, OmpDirective::SIMD,
    OmpDirective::TARGET_PARALLEL_DO_SIMD,
    OmpDirective::TARGET_TEAMS_DISTRIBUTE_PARALLEL_DO_SIMD,
    OmpDirective::TARGET_TEAMS_DISTRIBUTE_SIMD, OmpDirective::TARGET_SIMD,
    OmpDirective::TASKLOOP_SIMD,
    OmpDirective::TEAMS_DISTRIBUTE_PARALLEL_DO_SIMD,
    OmpDirective::TEAMS_DISTRIBUTE_SIMD};
static constexpr OmpDirectiveSet taskGeneratingSet{
    OmpDirectiveSet{OmpDirective::TASK} | taskloopSet};

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
  struct OmpContext {
    OmpContext(parser::CharBlock source, OmpDirective d)
      : directiveSource{source}, directive{d} {}
    parser::CharBlock directiveSource{nullptr};
    parser::CharBlock clauseSource{nullptr};
    OmpDirective directive;
    OmpClauseSet allowedClauses{};
    OmpClauseSet allowedOnceClauses{};
    OmpClauseSet allowedExclusiveClauses{};
    OmpClauseSet requiredClauses{};

    const parser::OmpClause *clause{nullptr};
    std::multimap<OmpClause, const parser::OmpClause *> clauseInfo;
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
  void SetContextDirectiveEnum(OmpDirective dir) {
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
  void SetContextClauseInfo(OmpClause type) {
    GetContext().clauseInfo.emplace(type, GetContext().clause);
  }
  const parser::OmpClause *FindClause(OmpClause type) {
    auto it{GetContext().clauseInfo.find(type)};
    if (it != GetContext().clauseInfo.end()) {
      return it->second;
    }
    return nullptr;
  }
  void PushContext(const parser::CharBlock &source, OmpDirective dir) {
    ompContext_.emplace_back(source, dir);
  }

  void RequiresConstantPositiveParameter(
      const OmpClause &clause, const parser::ScalarIntConstantExpr &i);
  void RequiresPositiveParameter(
      const OmpClause &clause, const parser::ScalarIntExpr &i);

  bool CurrentDirectiveIsNested() { return ompContext_.size() > 0; };
  bool HasInvalidWorksharingNesting(
      const parser::CharBlock &, const OmpDirectiveSet &);
  void CheckAllowed(OmpClause);
  void CheckRequired(OmpClause);
  std::string ContextDirectiveAsFortran();
  void SayNotMatching(const parser::CharBlock &, const parser::CharBlock &);
  template<typename A, typename B, typename C>
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
  std::vector<OmpContext> ompContext_;  // used as a stack
};
}
#endif  // FORTRAN_SEMANTICS_CHECK_OMP_STRUCTURE_H_
