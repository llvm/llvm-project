//===-- lib/Semantics/check-omp-structure.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-omp-structure.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"
#include <unordered_map>

namespace Fortran::semantics {

static OmpClauseSet doAllowedClauses{llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_firstprivate, llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_linear, llvm::omp::Clause::OMPC_reduction};
static OmpClauseSet doAllowedOnceClauses{llvm::omp::Clause::OMPC_schedule,
    llvm::omp::Clause::OMPC_collapse, llvm::omp::Clause::OMPC_ordered};

static OmpClauseSet simdAllowedClauses{llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_aligned, llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_lastprivate, llvm::omp::Clause::OMPC_reduction};
static OmpClauseSet simdAllowedOnceClauses{llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_safelen, llvm::omp::Clause::OMPC_simdlen};

static OmpClauseSet parallelAllowedClauses{llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_private, llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_shared, llvm::omp::Clause::OMPC_copyin,
    llvm::omp::Clause::OMPC_reduction};
static OmpClauseSet parallelAllowedOnceClauses{llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_num_threads, llvm::omp::Clause::OMPC_proc_bind};

static OmpClauseSet taskloopAllowedClauses{llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_private, llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_lastprivate, llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_untied, llvm::omp::Clause::OMPC_mergeable,
    llvm::omp::Clause::OMPC_nogroup};
static OmpClauseSet taskloopAllowedOnceClauses{llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_if, llvm::omp::Clause::OMPC_final,
    llvm::omp::Clause::OMPC_priority};
static OmpClauseSet taskloopAllowedExclusiveClauses{
    llvm::omp::Clause::OMPC_grainsize, llvm::omp::Clause::OMPC_num_tasks};

static OmpClauseSet distributeAllowedClauses{llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_firstprivate, llvm::omp::Clause::OMPC_lastprivate};
static OmpClauseSet distributeAllowedOnceClauses{
    llvm::omp::Clause::OMPC_collapse, llvm::omp::Clause::OMPC_dist_schedule};

static OmpClauseSet targetAllowedClauses{llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_private, llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_map, llvm::omp::Clause::OMPC_is_device_ptr,
    llvm::omp::Clause::OMPC_depend};
static OmpClauseSet targetAllowedOnceClauses{llvm::omp::Clause::OMPC_device,
    llvm::omp::Clause::OMPC_defaultmap, llvm::omp::Clause::OMPC_nowait};

static OmpClauseSet teamsAllowedClauses{llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_firstprivate, llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_reduction};
static OmpClauseSet teamsAllowedOnceClauses{llvm::omp::Clause::OMPC_num_teams,
    llvm::omp::Clause::OMPC_thread_limit, llvm::omp::Clause::OMPC_default};

static OmpClauseSet sectionsAllowedClauses{llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_firstprivate, llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_reduction};

std::string OmpStructureChecker::ContextDirectiveAsFortran() {
  auto dir = llvm::omp::getOpenMPDirectiveName(GetContext().directive).str();
  std::transform(dir.begin(), dir.end(), dir.begin(),
      [](unsigned char c) { return std::toupper(c); });
  return dir;
}

void OmpStructureChecker::SayNotMatching(
    const parser::CharBlock &beginSource, const parser::CharBlock &endSource) {
  context_
      .Say(endSource, "Unmatched %s directive"_err_en_US,
          parser::ToUpperCaseLetters(endSource.ToString()))
      .Attach(beginSource, "Does not match directive"_en_US);
}

bool OmpStructureChecker::HasInvalidWorksharingNesting(
    const parser::CharBlock &source, const OmpDirectiveSet &set) {
  // set contains all the invalid closely nested directives
  // for the given directive (`source` here)
  if (CurrentDirectiveIsNested() && set.test(GetContext().directive)) {
    context_.Say(source,
        "A worksharing region may not be closely nested inside a "
        "worksharing, explicit task, taskloop, critical, ordered, atomic, or "
        "master region"_err_en_US);
    return true;
  }
  return false;
}

void OmpStructureChecker::CheckAllowed(llvm::omp::Clause type) {
  if (!GetContext().allowedClauses.test(type) &&
      !GetContext().allowedOnceClauses.test(type) &&
      !GetContext().allowedExclusiveClauses.test(type)) {
    context_.Say(GetContext().clauseSource,
        "%s clause is not allowed on the %s directive"_err_en_US,
        parser::ToUpperCaseLetters(llvm::omp::getOpenMPClauseName(type).str()),
        parser::ToUpperCaseLetters(GetContext().directiveSource.ToString()));
    return;
  }
  if ((GetContext().allowedOnceClauses.test(type) ||
          GetContext().allowedExclusiveClauses.test(type)) &&
      FindClause(type)) {
    context_.Say(GetContext().clauseSource,
        "At most one %s clause can appear on the %s directive"_err_en_US,
        parser::ToUpperCaseLetters(llvm::omp::getOpenMPClauseName(type).str()),
        parser::ToUpperCaseLetters(GetContext().directiveSource.ToString()));
    return;
  }
  if (GetContext().allowedExclusiveClauses.test(type)) {
    std::vector<llvm::omp::Clause> others;
    GetContext().allowedExclusiveClauses.IterateOverMembers(
        [&](llvm::omp::Clause o) {
          if (FindClause(o)) {
            others.emplace_back(o);
          }
        });
    for (const auto &e : others) {
      context_.Say(GetContext().clauseSource,
          "%s and %s are mutually exclusive and may not appear on the "
          "same %s directive"_err_en_US,
          parser::ToUpperCaseLetters(
              llvm::omp::getOpenMPClauseName(type).str()),
          parser::ToUpperCaseLetters(llvm::omp::getOpenMPClauseName(e).str()),
          parser::ToUpperCaseLetters(GetContext().directiveSource.ToString()));
    }
    if (!others.empty()) {
      return;
    }
  }
  SetContextClauseInfo(type);
}

void OmpStructureChecker::CheckRequired(llvm::omp::Clause c) {
  if (!FindClause(c)) {
    context_.Say(GetContext().directiveSource,
        "At least one %s clause must appear on the %s directive"_err_en_US,
        parser::ToUpperCaseLetters(llvm::omp::getOpenMPClauseName(c).str()),
        ContextDirectiveAsFortran());
  }
}

void OmpStructureChecker::RequiresConstantPositiveParameter(
    const llvm::omp::Clause &clause, const parser::ScalarIntConstantExpr &i) {
  if (const auto v{GetIntValue(i)}) {
    if (*v <= 0) {
      context_.Say(GetContext().clauseSource,
          "The parameter of the %s clause must be "
          "a constant positive integer expression"_err_en_US,
          parser::ToUpperCaseLetters(
              llvm::omp::getOpenMPClauseName(clause).str()));
    }
  }
}

void OmpStructureChecker::RequiresPositiveParameter(
    const llvm::omp::Clause &clause, const parser::ScalarIntExpr &i) {
  if (const auto v{GetIntValue(i)}) {
    if (*v <= 0) {
      context_.Say(GetContext().clauseSource,
          "The parameter of the %s clause must be "
          "a positive integer expression"_err_en_US,
          parser::ToUpperCaseLetters(
              llvm::omp::getOpenMPClauseName(clause).str()));
    }
  }
}

void OmpStructureChecker::Enter(const parser::OpenMPConstruct &) {
  // 2.8.1 TODO: Simd Construct with Ordered Construct Nesting check
}

void OmpStructureChecker::Enter(const parser::OpenMPLoopConstruct &x) {
  const auto &beginLoopDir{std::get<parser::OmpBeginLoopDirective>(x.t)};
  const auto &beginDir{std::get<parser::OmpLoopDirective>(beginLoopDir.t)};

  // check matching, End directive is optional
  if (const auto &endLoopDir{
          std::get<std::optional<parser::OmpEndLoopDirective>>(x.t)}) {
    CheckMatching<parser::OmpLoopDirective>(beginLoopDir, *endLoopDir);
  }

  if (beginDir.v != llvm::omp::Directive::OMPD_do)
    PushContext(beginDir.source, beginDir.v);

  switch (beginDir.v) {
  // 2.7.1 do-clause -> private-clause |
  //                    firstprivate-clause |
  //                    lastprivate-clause |
  //                    linear-clause |
  //                    reduction-clause |
  //                    schedule-clause |
  //                    collapse-clause |
  //                    ordered-clause
  case llvm::omp::Directive::OMPD_do: {
    // nesting check
    HasInvalidWorksharingNesting(beginDir.source,
        {llvm::omp::Directive::OMPD_do, llvm::omp::Directive::OMPD_sections,
            llvm::omp::Directive::OMPD_single,
            llvm::omp::Directive::OMPD_workshare,
            llvm::omp::Directive::OMPD_task,
            llvm::omp::Directive::OMPD_taskloop,
            llvm::omp::Directive::OMPD_critical,
            llvm::omp::Directive::OMPD_ordered,
            llvm::omp::Directive::OMPD_atomic,
            llvm::omp::Directive::OMPD_master});
    PushContext(beginDir.source, llvm::omp::Directive::OMPD_do);
    SetContextAllowed(doAllowedClauses);
    SetContextAllowedOnce(doAllowedOnceClauses);
  } break;

  // 2.11.1 parallel-do-clause -> parallel-clause |
  //                              do-clause
  case llvm::omp::Directive::OMPD_parallel_do: {
    SetContextAllowed(parallelAllowedClauses | doAllowedClauses);
    SetContextAllowedOnce(parallelAllowedOnceClauses | doAllowedOnceClauses);
  } break;

  // 2.8.1 simd-clause -> safelen-clause |
  //                      simdlen-clause |
  //                      linear-clause |
  //                      aligned-clause |
  //                      private-clause |
  //                      lastprivate-clause |
  //                      reduction-clause |
  //                      collapse-clause
  case llvm::omp::Directive::OMPD_simd: {
    SetContextAllowed(simdAllowedClauses);
    SetContextAllowedOnce(simdAllowedOnceClauses);
  } break;

  // 2.8.3 do-simd-clause -> do-clause |
  //                         simd-clause
  case llvm::omp::Directive::OMPD_do_simd: {
    SetContextAllowed(doAllowedClauses | simdAllowedClauses);
    SetContextAllowedOnce(doAllowedOnceClauses | simdAllowedOnceClauses);
  } break;

  // 2.11.4 parallel-do-simd-clause -> parallel-clause |
  //                                   do-simd-clause
  case llvm::omp::Directive::OMPD_parallel_do_simd: {
    SetContextAllowed(
        parallelAllowedClauses | doAllowedClauses | simdAllowedClauses);
    SetContextAllowedOnce(parallelAllowedOnceClauses | doAllowedOnceClauses |
        simdAllowedOnceClauses);
  } break;

  // 2.9.2 taskloop-clause -> if-clause |
  //                          shared-clause |
  //                          private-clause |
  //                          firstprivate-clause |
  //                          lastprivate-clause |
  //                          default-clause |
  //                          grainsize-clause |
  //                          num-tasks-clause |
  //                          collapse-clause |
  //                          final-clause |
  //                          priority-clause |
  //                          untied-clause |
  //                          mergeable-clause |
  //                          nogroup-clause
  case llvm::omp::Directive::OMPD_taskloop: {
    SetContextAllowed(taskloopAllowedClauses);
    SetContextAllowedOnce(taskloopAllowedOnceClauses);
    SetContextAllowedExclusive(taskloopAllowedExclusiveClauses);
  } break;

  // 2.9.3 taskloop-simd-clause -> taskloop-clause |
  //                               simd-clause
  case llvm::omp::Directive::OMPD_taskloop_simd: {
    SetContextAllowed((taskloopAllowedClauses | simdAllowedClauses) -
        llvm::omp::Clause::OMPC_reduction);
    SetContextAllowedOnce(taskloopAllowedOnceClauses | simdAllowedOnceClauses);
    SetContextAllowedExclusive(taskloopAllowedExclusiveClauses);
  } break;

  // 2.10.8 distribute-clause -> private-clause |
  //                             firstprivate-clause |
  //                             lastprivate-clause |
  //                             collapse-clause |
  //                             dist-schedule-clause
  case llvm::omp::Directive::OMPD_distribute: {
    SetContextAllowed(distributeAllowedClauses);
    SetContextAllowedOnce(distributeAllowedOnceClauses);
  } break;

  // 2.10.9 distribute-simd-clause -> distribute-clause |
  //                                  simd-clause
  case llvm::omp::Directive::OMPD_distribute_simd: {
    SetContextAllowed(distributeAllowedClauses | simdAllowedClauses);
    SetContextAllowedOnce(
        distributeAllowedOnceClauses | simdAllowedOnceClauses);
  } break;

  // 2.10.10 distribute-parallel-do-clause -> distribute-clause |
  //                                          parallel-do-clause
  case llvm::omp::Directive::OMPD_distribute_parallel_do: {
    SetContextAllowed(
        distributeAllowedClauses | parallelAllowedClauses | doAllowedClauses);
    SetContextAllowedOnce(distributeAllowedOnceClauses |
        parallelAllowedOnceClauses | doAllowedOnceClauses);
  } break;

  // 2.10.11 distribute-parallel-do-simd-clause -> distribute-clause |
  //                                               parallel-do-simd-clause
  case llvm::omp::Directive::OMPD_distribute_parallel_do_simd: {
    SetContextAllowed(distributeAllowedClauses | parallelAllowedClauses |
        doAllowedClauses | simdAllowedClauses);
    SetContextAllowedOnce(distributeAllowedOnceClauses |
        parallelAllowedOnceClauses | doAllowedOnceClauses | simdAllowedClauses);
  } break;

  // 2.11.6 target-parallel-do-clause -> target-clause |
  //                                     parallel-do-clause
  case llvm::omp::Directive::OMPD_target_parallel_do: {
    SetContextAllowed(
        targetAllowedClauses | parallelAllowedClauses | doAllowedClauses);
    SetContextAllowedOnce(
        (targetAllowedOnceClauses | parallelAllowedOnceClauses |
            doAllowedOnceClauses) -
        llvm::omp::Clause::OMPC_nowait);
  } break;

  // 2.11.7 target-parallel-do-simd-clause -> target-clause |
  //                                          parallel-do-simd-clause
  case llvm::omp::Directive::OMPD_target_parallel_do_simd: {
    SetContextAllowed(targetAllowedClauses | parallelAllowedClauses |
        doAllowedClauses | simdAllowedClauses);
    SetContextAllowedOnce(
        (targetAllowedOnceClauses | parallelAllowedOnceClauses |
            doAllowedOnceClauses | simdAllowedOnceClauses) -
        llvm::omp::Clause::OMPC_nowait);
  } break;

  // 2.11.8 target-simd-clause -> target-clause |
  //                              simd-clause
  case llvm::omp::Directive::OMPD_target_simd: {
    SetContextAllowed(targetAllowedClauses | simdAllowedClauses);
    SetContextAllowedOnce(targetAllowedOnceClauses | simdAllowedOnceClauses);
  } break;

  // 2.11.10 teams-distribute-clause -> teams-clause |
  //                                    distribute-clause
  case llvm::omp::Directive::OMPD_teams_distribute: {
    SetContextAllowed(teamsAllowedClauses | distributeAllowedClauses);
    SetContextAllowedOnce(
        teamsAllowedOnceClauses | distributeAllowedOnceClauses);
  } break;

  // 2.11.11 teams-distribute-simd-clause -> teams-clause |
  //                                         distribute-simd-clause
  case llvm::omp::Directive::OMPD_teams_distribute_simd: {
    SetContextAllowed(
        teamsAllowedClauses | distributeAllowedClauses | simdAllowedClauses);
    SetContextAllowedOnce(teamsAllowedOnceClauses |
        distributeAllowedOnceClauses | simdAllowedOnceClauses);
  } break;

  // 2.11.12 target-teams-distribute-clause -> target-clause |
  //                                           teams-distribute-clause
  case llvm::omp::Directive::OMPD_target_teams_distribute: {
    SetContextAllowed(
        targetAllowedClauses | teamsAllowedClauses | distributeAllowedClauses);
    SetContextAllowedOnce(targetAllowedOnceClauses | teamsAllowedOnceClauses |
        distributeAllowedOnceClauses);
  } break;

  // 2.11.13 target-teams-distribute-simd-clause -> target-clause |
  //                                                teams-distribute-simd-clause
  case llvm::omp::Directive::OMPD_target_teams_distribute_simd: {
    SetContextAllowed(targetAllowedClauses | teamsAllowedClauses |
        distributeAllowedClauses | simdAllowedClauses);
    SetContextAllowed(targetAllowedOnceClauses | teamsAllowedOnceClauses |
        distributeAllowedOnceClauses | simdAllowedOnceClauses);
  } break;

  // 2.11.14 teams-distribute-parallel-do-clause -> teams-clause |
  //                                                distribute-parallel-do-clause
  case llvm::omp::Directive::OMPD_teams_distribute_parallel_do: {
    SetContextAllowed(teamsAllowedClauses | distributeAllowedClauses |
        parallelAllowedClauses | doAllowedClauses);
    SetContextAllowedOnce(teamsAllowedOnceClauses |
        distributeAllowedOnceClauses | parallelAllowedOnceClauses |
        doAllowedOnceClauses);
  } break;

  // 2.11.15 target-teams-distribute-parallel-do-clause -> target-clause |
  //                                                       teams-distribute-parallel-do-clause
  case llvm::omp::Directive::OMPD_target_teams_distribute_parallel_do: {
    SetContextAllowed(targetAllowedClauses | teamsAllowedClauses |
        distributeAllowedClauses | parallelAllowedClauses | doAllowedClauses);
    SetContextAllowedOnce(targetAllowedOnceClauses | teamsAllowedOnceClauses |
        distributeAllowedOnceClauses | parallelAllowedOnceClauses |
        doAllowedOnceClauses);
  } break;

  // 2.11.16 teams-distribute-parallel-do-clause -> teams-clause |
  //                                                distribute-parallel-do-simd-clause
  case llvm::omp::Directive::OMPD_teams_distribute_parallel_do_simd: {
    SetContextAllowed(teamsAllowedClauses | distributeAllowedClauses |
        parallelAllowedClauses | doAllowedClauses | simdAllowedClauses);
    SetContextAllowedOnce(teamsAllowedOnceClauses |
        distributeAllowedOnceClauses | parallelAllowedOnceClauses |
        doAllowedOnceClauses | simdAllowedOnceClauses);
  } break;

  case llvm::omp::Directive::OMPD_target_teams_distribute_parallel_do_simd: {
    SetContextAllowed(targetAllowedClauses | teamsAllowedClauses |
        distributeAllowedClauses | parallelAllowedClauses | doAllowedClauses |
        simdAllowedClauses);
    SetContextAllowedOnce(targetAllowedOnceClauses | teamsAllowedOnceClauses |
        distributeAllowedOnceClauses | parallelAllowedOnceClauses |
        doAllowedOnceClauses | simdAllowedOnceClauses);
  } break;

  default:
    // TODO others
    break;
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPLoopConstruct &) {
  ompContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OmpEndLoopDirective &x) {
  const auto &dir{std::get<parser::OmpLoopDirective>(x.t)};
  ResetPartialContext(dir.source);
  switch (dir.v) {
  // 2.7.1 end-do -> END DO [nowait-clause]
  // 2.8.3 end-do-simd -> END DO SIMD [nowait-clause]
  case llvm::omp::Directive::OMPD_do:
    SetContextDirectiveEnum(llvm::omp::Directive::OMPD_end_do);
    SetContextAllowed(OmpClauseSet{llvm::omp::Clause::OMPC_nowait});
    break;
  case llvm::omp::Directive::OMPD_do_simd:
    SetContextDirectiveEnum(llvm::omp::Directive::OMPD_end_do_simd);
    SetContextAllowed(OmpClauseSet{llvm::omp::Clause::OMPC_nowait});
    break;
  default:
    // no clauses are allowed
    break;
  }
}

void OmpStructureChecker::Enter(const parser::OpenMPBlockConstruct &x) {
  const auto &beginBlockDir{std::get<parser::OmpBeginBlockDirective>(x.t)};
  const auto &endBlockDir{std::get<parser::OmpEndBlockDirective>(x.t)};
  const auto &beginDir{
      CheckMatching<parser::OmpBlockDirective>(beginBlockDir, endBlockDir)};

  PushContext(beginDir.source, beginDir.v);
  switch (beginDir.v) {
  // 2.5 parallel-clause -> if-clause |
  //                        num-threads-clause |
  //                        default-clause |
  //                        private-clause |
  //                        firstprivate-clause |
  //                        shared-clause |
  //                        copyin-clause |
  //                        reduction-clause |
  //                        proc-bind-clause
  case llvm::omp::Directive::OMPD_parallel: {
    // reserve for nesting check
    SetContextAllowed(parallelAllowedClauses);
    SetContextAllowedOnce(parallelAllowedOnceClauses);
  } break;
  // 2.7.3 single-clause -> private-clause |
  //                        firstprivate-clause
  case llvm::omp::Directive::OMPD_single:
    SetContextAllowed({llvm::omp::Clause::OMPC_private,
        llvm::omp::Clause::OMPC_firstprivate});
    break;
  // 2.7.4 workshare (no clauses are allowed)
  case llvm::omp::Directive::OMPD_workshare:
    break;
  // 2.11.3 parallel-workshare-clause -> parallel-clause
  case llvm::omp::Directive::OMPD_parallel_workshare: {
    SetContextAllowed(parallelAllowedClauses);
    SetContextAllowedOnce(parallelAllowedOnceClauses);
  } break;
    // 2.9.1 task-clause -> if-clause |
    //                      final-clause |
    //                      untied-clause |
    //                      default-clause |
    //                      mergeable-clause |
    //                      private-clause |
    //                      firstprivate-clause |
    //                      shared-clause |
    //                      depend-clause |
    //                      priority-clause
  case llvm::omp::Directive::OMPD_task: {
    OmpClauseSet allowed{llvm::omp::Clause::OMPC_untied,
        llvm::omp::Clause::OMPC_default, llvm::omp::Clause::OMPC_mergeable,
        llvm::omp::Clause::OMPC_private, llvm::omp::Clause::OMPC_firstprivate,
        llvm::omp::Clause::OMPC_shared, llvm::omp::Clause::OMPC_depend};
    SetContextAllowed(allowed);
    OmpClauseSet allowedOnce{llvm::omp::Clause::OMPC_if,
        llvm::omp::Clause::OMPC_final, llvm::omp::Clause::OMPC_priority};
    SetContextAllowedOnce(allowedOnce);
  } break;
  // 2.10.4 target-clause -> if-clause |
  //                         device-clause |
  //                         private-clause |
  //                         firstprivate-clause |
  //                         map-clause |
  //                         is-device-ptr-clause |
  //                         defaultmap-clause |
  //                         nowait-clause |
  //                         depend-clause
  case llvm::omp::Directive::OMPD_target: {
    SetContextAllowed(targetAllowedClauses);
    SetContextAllowedOnce(targetAllowedOnceClauses);
  } break;
  // 2.10.7 teams-clause -> num-teams-clause |
  //                        thread-limit-clause |
  //                        default-clause |
  //                        private-clause |
  //                        firstprivate-clause |
  //                        shared-clause |
  //                        reduction-clause
  case llvm::omp::Directive::OMPD_teams: {
    SetContextAllowed(teamsAllowedClauses);
    SetContextAllowedOnce(teamsAllowedOnceClauses);
  } break;
  // 2.11.9 target-teams -> target-clause |
  //                        teams-clause
  case llvm::omp::Directive::OMPD_target_teams: {
    SetContextAllowed(targetAllowedClauses | teamsAllowedClauses);
    SetContextAllowedOnce(targetAllowedOnceClauses | teamsAllowedOnceClauses);
  } break;
  // 2.10.1 target-data-clause -> if-clause |
  //                              device-clause |
  //                              map-clause |
  //                              use-device-ptr-clause
  case llvm::omp::Directive::OMPD_target_data: {
    OmpClauseSet allowed{llvm::omp::Clause::OMPC_if,
        llvm::omp::Clause::OMPC_map, llvm::omp::Clause::OMPC_use_device_ptr};
    SetContextAllowed(allowed);
    SetContextAllowedOnce({llvm::omp::Clause::OMPC_device});
    SetContextRequired({llvm::omp::Clause::OMPC_map});
  } break;
  // 2.13.1 master (no clauses are allowed)
  case llvm::omp::Directive::OMPD_master:
    break;
  // 2.11.5 target-parallel-clause -> target-clause |
  //                                  parallel-clause
  case llvm::omp::Directive::OMPD_target_parallel: {
    SetContextAllowed((targetAllowedClauses | parallelAllowedClauses) -
        llvm::omp::Clause::OMPC_copyin);
    SetContextAllowedOnce(
        targetAllowedOnceClauses | parallelAllowedOnceClauses);
  } break;
  default:
    // TODO others
    break;
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPBlockConstruct &) {
  ompContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPSectionsConstruct &x) {
  const auto &beginSectionsDir{
      std::get<parser::OmpBeginSectionsDirective>(x.t)};
  const auto &endSectionsDir{std::get<parser::OmpEndSectionsDirective>(x.t)};
  const auto &beginDir{CheckMatching<parser::OmpSectionsDirective>(
      beginSectionsDir, endSectionsDir)};

  PushContext(beginDir.source, beginDir.v);
  switch (beginDir.v) {
  // 2.7.2 sections-clause -> private-clause |
  //                          firstprivate-clause |
  //                          lastprivate-clause |
  //                          reduction-clause
  case llvm::omp::Directive::OMPD_sections: {
    SetContextAllowed(sectionsAllowedClauses);
  } break;
    // 2.11.2 -> parallel-sections-clause -> parallel-clause |
    //                                       sections-clause
  case llvm::omp::Directive::OMPD_parallel_sections: {
    SetContextAllowed(parallelAllowedClauses | sectionsAllowedClauses);
    SetContextAllowedOnce(parallelAllowedOnceClauses);
  } break;
  default:
    // TODO others
    break;
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPSectionsConstruct &) {
  ompContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OmpEndSectionsDirective &x) {
  const auto &dir{std::get<parser::OmpSectionsDirective>(x.t)};
  ResetPartialContext(dir.source);
  switch (dir.v) {
    // 2.7.2 end-sections -> END SECTIONS [nowait-clause]
  case llvm::omp::Directive::OMPD_sections:
    SetContextDirectiveEnum(llvm::omp::Directive::OMPD_end_sections);
    SetContextAllowed(OmpClauseSet{llvm::omp::Clause::OMPC_nowait});
    break;
  default:
    // no clauses are allowed
    break;
  }
}

void OmpStructureChecker::Enter(const parser::OpenMPDeclareSimdConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContext(dir.source, llvm::omp::Directive::OMPD_declare_simd);
  // 2.8.2 declare-simd-clause -> simdlen-clause |
  //                              linear-clause |
  //                              aligned-clause |
  //                              uniform-clause |
  //                              inbranch-clause |
  //                              notinbranch-clause
  OmpClauseSet allowed{llvm::omp::Clause::OMPC_linear,
      llvm::omp::Clause::OMPC_aligned, llvm::omp::Clause::OMPC_uniform};
  SetContextAllowed(allowed);
  SetContextAllowedOnce({llvm::omp::Clause::OMPC_simdlen});
  SetContextAllowedExclusive(
      {llvm::omp::Clause::OMPC_inbranch, llvm::omp::Clause::OMPC_notinbranch});
}

void OmpStructureChecker::Leave(const parser::OpenMPDeclareSimdConstruct &) {
  ompContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPDeclareTargetConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContext(dir.source, llvm::omp::Directive::OMPD_declare_target);
  const auto &spec{std::get<parser::OmpDeclareTargetSpecifier>(x.t)};
  if (std::holds_alternative<parser::OmpDeclareTargetWithClause>(spec.u)) {
    SetContextAllowed(
        OmpClauseSet{llvm::omp::Clause::OMPC_to, llvm::omp::Clause::OMPC_link});
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPDeclareTargetConstruct &) {
  ompContext_.pop_back();
}

void OmpStructureChecker::Enter(
    const parser::OpenMPSimpleStandaloneConstruct &x) {
  const auto &dir{std::get<parser::OmpSimpleStandaloneDirective>(x.t)};
  PushContext(dir.source, dir.v);
  switch (dir.v) {
  case llvm::omp::Directive::OMPD_barrier: {
    // 2.13.3 barrier
  } break;
  case llvm::omp::Directive::OMPD_taskwait: {
    // 2.13.4 taskwait
  } break;
  case llvm::omp::Directive::OMPD_taskyield: {
    // 2.9.4 taskyield
  } break;
  case llvm::omp::Directive::OMPD_target_enter_data: {
    // 2.10.2 target-enter-data-clause -> if-clause |
    //                                    device-clause |
    //                                    map-clause |
    //                                    depend-clause |
    //                                    nowait-clause
    OmpClauseSet allowed{llvm::omp::Clause::OMPC_map,
        llvm::omp::Clause::OMPC_depend, llvm::omp::Clause::OMPC_nowait};
    SetContextAllowed(allowed);
    OmpClauseSet allowedOnce{
        llvm::omp::Clause::OMPC_device, llvm::omp::Clause::OMPC_if};
    SetContextAllowedOnce(allowedOnce);
    SetContextRequired({llvm::omp::Clause::OMPC_map});
  } break;
  case llvm::omp::Directive::OMPD_target_exit_data: {
    // 2.10.3  target-enter-data-clause -> if-clause |
    //                                     device-clause |
    //                                     map-clause |
    //                                     depend-clause |
    //                                     nowait-clause
    OmpClauseSet allowed{llvm::omp::Clause::OMPC_map,
        llvm::omp::Clause::OMPC_depend, llvm::omp::Clause::OMPC_nowait};
    SetContextAllowed(allowed);
    OmpClauseSet allowedOnce{
        llvm::omp::Clause::OMPC_device, llvm::omp::Clause::OMPC_if};
    SetContextAllowedOnce(allowedOnce);
    SetContextRequired({llvm::omp::Clause::OMPC_map});
  } break;
  case llvm::omp::Directive::OMPD_target_update: {
    // 2.10.5 target-update
  } break;
  case llvm::omp::Directive::OMPD_ordered: {
    // 2.13.8 ordered-construct-clause -> depend-clause
    OmpClauseSet allowed{llvm::omp::Clause::OMPC_depend};
    SetContextAllowed(allowed);
  } break;
  default:
    // TODO others
    break;
  }
}

void OmpStructureChecker::Leave(
    const parser::OpenMPSimpleStandaloneConstruct &) {
  ompContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPFlushConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContext(dir.source, llvm::omp::Directive::OMPD_flush);
}

void OmpStructureChecker::Leave(const parser::OpenMPFlushConstruct &) {
  ompContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPCancelConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContext(dir.source, llvm::omp::Directive::OMPD_cancel);
}

void OmpStructureChecker::Leave(const parser::OpenMPCancelConstruct &) {
  ompContext_.pop_back();
}

void OmpStructureChecker::Enter(
    const parser::OpenMPCancellationPointConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContext(dir.source, llvm::omp::Directive::OMPD_cancellation_point);
}

void OmpStructureChecker::Leave(
    const parser::OpenMPCancellationPointConstruct &) {
  ompContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OmpEndBlockDirective &x) {
  const auto &dir{std::get<parser::OmpBlockDirective>(x.t)};
  ResetPartialContext(dir.source);
  switch (dir.v) {
  // 2.7.3 end-single-clause -> copyprivate-clause |
  //                            nowait-clause
  case llvm::omp::Directive::OMPD_single: {
    SetContextDirectiveEnum(llvm::omp::Directive::OMPD_end_single);
    OmpClauseSet allowed{llvm::omp::Clause::OMPC_copyprivate};
    SetContextAllowed(allowed);
    OmpClauseSet allowedOnce{llvm::omp::Clause::OMPC_nowait};
    SetContextAllowedOnce(allowedOnce);
  } break;
  // 2.7.4 end-workshare -> END WORKSHARE [nowait-clause]
  case llvm::omp::Directive::OMPD_workshare:
    SetContextDirectiveEnum(llvm::omp::Directive::OMPD_end_workshare);
    SetContextAllowed(OmpClauseSet{llvm::omp::Clause::OMPC_nowait});
    break;
  default:
    // no clauses are allowed
    break;
  }
}

void OmpStructureChecker::Leave(const parser::OmpClauseList &) {
  // 2.7 Loop Construct Restriction
  if (llvm::omp::doSet.test(GetContext().directive)) {
    if (auto *clause{FindClause(llvm::omp::Clause::OMPC_schedule)}) {
      // only one schedule clause is allowed
      const auto &schedClause{std::get<parser::OmpScheduleClause>(clause->u)};
      if (ScheduleModifierHasType(schedClause,
              parser::OmpScheduleModifierType::ModType::Nonmonotonic)) {
        if (FindClause(llvm::omp::Clause::OMPC_ordered)) {
          context_.Say(clause->source,
              "The NONMONOTONIC modifier cannot be specified "
              "if an ORDERED clause is specified"_err_en_US);
        }
        if (ScheduleModifierHasType(schedClause,
                parser::OmpScheduleModifierType::ModType::Monotonic)) {
          context_.Say(clause->source,
              "The MONOTONIC and NONMONOTONIC modifiers "
              "cannot be both specified"_err_en_US);
        }
      }
    }

    if (auto *clause{FindClause(llvm::omp::Clause::OMPC_ordered)}) {
      // only one ordered clause is allowed
      const auto &orderedClause{
          std::get<parser::OmpClause::Ordered>(clause->u)};

      if (orderedClause.v) {
        if (FindClause(llvm::omp::Clause::OMPC_linear)) {
          context_.Say(clause->source,
              "A loop directive may not have both a LINEAR clause and "
              "an ORDERED clause with a parameter"_err_en_US);
        }

        if (auto *clause2{FindClause(llvm::omp::Clause::OMPC_collapse)}) {
          const auto &collapseClause{
              std::get<parser::OmpClause::Collapse>(clause2->u)};
          // ordered and collapse both have parameters
          if (const auto orderedValue{GetIntValue(orderedClause.v)}) {
            if (const auto collapseValue{GetIntValue(collapseClause.v)}) {
              if (*orderedValue > 0 && *orderedValue < *collapseValue) {
                context_.Say(clause->source,
                    "The parameter of the ORDERED clause must be "
                    "greater than or equal to "
                    "the parameter of the COLLAPSE clause"_err_en_US);
              }
            }
          }
        }
      }

      // TODO: ordered region binding check (requires nesting implementation)
    }
  } // doSet

  // 2.8.1 Simd Construct Restriction
  if (llvm::omp::simdSet.test(GetContext().directive)) {
    if (auto *clause{FindClause(llvm::omp::Clause::OMPC_simdlen)}) {
      if (auto *clause2{FindClause(llvm::omp::Clause::OMPC_safelen)}) {
        const auto &simdlenClause{
            std::get<parser::OmpClause::Simdlen>(clause->u)};
        const auto &safelenClause{
            std::get<parser::OmpClause::Safelen>(clause2->u)};
        // simdlen and safelen both have parameters
        if (const auto simdlenValue{GetIntValue(simdlenClause.v)}) {
          if (const auto safelenValue{GetIntValue(safelenClause.v)}) {
            if (*safelenValue > 0 && *simdlenValue > *safelenValue) {
              context_.Say(clause->source,
                  "The parameter of the SIMDLEN clause must be less than or "
                  "equal to the parameter of the SAFELEN clause"_err_en_US);
            }
          }
        }
      }
    }

    // TODO: A list-item cannot appear in more than one aligned clause
  } // SIMD

  // 2.7.3 Single Construct Restriction
  if (GetContext().directive == llvm::omp::Directive::OMPD_end_single) {
    if (auto *clause{FindClause(llvm::omp::Clause::OMPC_copyprivate)}) {
      if (FindClause(llvm::omp::Clause::OMPC_nowait)) {
        context_.Say(clause->source,
            "The COPYPRIVATE clause must not be used with "
            "the NOWAIT clause"_err_en_US);
      }
    }
  }

  GetContext().requiredClauses.IterateOverMembers(
      [this](llvm::omp::Clause c) { CheckRequired(c); });
}

void OmpStructureChecker::Enter(const parser::OmpClause &x) {
  SetContextClause(x);
}

void OmpStructureChecker::Enter(const parser::OmpNowait &) {
  CheckAllowed(llvm::omp::Clause::OMPC_nowait);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Inbranch &) {
  CheckAllowed(llvm::omp::Clause::OMPC_inbranch);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Mergeable &) {
  CheckAllowed(llvm::omp::Clause::OMPC_mergeable);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Nogroup &) {
  CheckAllowed(llvm::omp::Clause::OMPC_nogroup);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Notinbranch &) {
  CheckAllowed(llvm::omp::Clause::OMPC_notinbranch);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Untied &) {
  CheckAllowed(llvm::omp::Clause::OMPC_untied);
}

void OmpStructureChecker::Enter(const parser::OmpClause::Collapse &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_collapse);
  // collapse clause must have a parameter
  RequiresConstantPositiveParameter(llvm::omp::Clause::OMPC_collapse, x.v);
}

void OmpStructureChecker::Enter(const parser::OmpClause::Copyin &) {
  CheckAllowed(llvm::omp::Clause::OMPC_copyin);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Copyprivate &) {
  CheckAllowed(llvm::omp::Clause::OMPC_copyprivate);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Device &) {
  CheckAllowed(llvm::omp::Clause::OMPC_device);
}
void OmpStructureChecker::Enter(const parser::OmpClause::DistSchedule &) {
  CheckAllowed(llvm::omp::Clause::OMPC_dist_schedule);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Final &) {
  CheckAllowed(llvm::omp::Clause::OMPC_final);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Firstprivate &) {
  CheckAllowed(llvm::omp::Clause::OMPC_firstprivate);
}
void OmpStructureChecker::Enter(const parser::OmpClause::From &) {
  CheckAllowed(llvm::omp::Clause::OMPC_from);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Grainsize &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_grainsize);
  RequiresPositiveParameter(llvm::omp::Clause::OMPC_grainsize, x.v);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Lastprivate &) {
  CheckAllowed(llvm::omp::Clause::OMPC_lastprivate);
}
void OmpStructureChecker::Enter(const parser::OmpClause::NumTasks &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_num_tasks);
  RequiresPositiveParameter(llvm::omp::Clause::OMPC_num_tasks, x.v);
}
void OmpStructureChecker::Enter(const parser::OmpClause::NumTeams &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_num_teams);
  RequiresPositiveParameter(llvm::omp::Clause::OMPC_num_teams, x.v);
}
void OmpStructureChecker::Enter(const parser::OmpClause::NumThreads &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_num_threads);
  RequiresPositiveParameter(llvm::omp::Clause::OMPC_num_threads, x.v);
  // if parameter is variable, defer to Expression Analysis
}

void OmpStructureChecker::Enter(const parser::OmpClause::Ordered &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_ordered);
  // the parameter of ordered clause is optional
  if (const auto &expr{x.v}) {
    RequiresConstantPositiveParameter(llvm::omp::Clause::OMPC_ordered, *expr);

    // 2.8.3 Loop SIMD Construct Restriction
    if (llvm::omp::doSimdSet.test(GetContext().directive)) {
      context_.Say(GetContext().clauseSource,
          "No ORDERED clause with a parameter can be specified "
          "on the %s directive"_err_en_US,
          ContextDirectiveAsFortran());
    }
  }
}
void OmpStructureChecker::Enter(const parser::OmpClause::Priority &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_priority);
  RequiresPositiveParameter(llvm::omp::Clause::OMPC_priority, x.v);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Private &) {
  CheckAllowed(llvm::omp::Clause::OMPC_private);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Safelen &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_safelen);
  RequiresConstantPositiveParameter(llvm::omp::Clause::OMPC_safelen, x.v);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Shared &) {
  CheckAllowed(llvm::omp::Clause::OMPC_shared);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Simdlen &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_simdlen);
  RequiresConstantPositiveParameter(llvm::omp::Clause::OMPC_simdlen, x.v);
}
void OmpStructureChecker::Enter(const parser::OmpClause::ThreadLimit &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_thread_limit);
  RequiresPositiveParameter(llvm::omp::Clause::OMPC_thread_limit, x.v);
}
void OmpStructureChecker::Enter(const parser::OmpClause::To &) {
  CheckAllowed(llvm::omp::Clause::OMPC_to);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Link &) {
  CheckAllowed(llvm::omp::Clause::OMPC_link);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Uniform &) {
  CheckAllowed(llvm::omp::Clause::OMPC_uniform);
}
void OmpStructureChecker::Enter(const parser::OmpClause::UseDevicePtr &) {
  CheckAllowed(llvm::omp::Clause::OMPC_use_device_ptr);
}
void OmpStructureChecker::Enter(const parser::OmpClause::IsDevicePtr &) {
  CheckAllowed(llvm::omp::Clause::OMPC_is_device_ptr);
}

void OmpStructureChecker::Enter(const parser::OmpAlignedClause &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_aligned);

  if (const auto &expr{
          std::get<std::optional<parser::ScalarIntConstantExpr>>(x.t)}) {
    if (const auto v{GetIntValue(*expr)}) {
      if (*v <= 0) {
        context_.Say(GetContext().clauseSource,
            "The ALIGNMENT parameter of the ALIGNED clause must be "
            "a constant positive integer expression"_err_en_US);
      }
    }
  }
  // 2.8.1 TODO: list-item attribute check
}
void OmpStructureChecker::Enter(const parser::OmpDefaultClause &) {
  CheckAllowed(llvm::omp::Clause::OMPC_default);
}
void OmpStructureChecker::Enter(const parser::OmpDefaultmapClause &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_defaultmap);
  using VariableCategory = parser::OmpDefaultmapClause::VariableCategory;
  if (!std::get<std::optional<VariableCategory>>(x.t)) {
    context_.Say(GetContext().clauseSource,
        "The argument TOFROM:SCALAR must be specified on the DEFAULTMAP "
        "clause"_err_en_US);
  }
}
void OmpStructureChecker::Enter(const parser::OmpDependClause &) {
  CheckAllowed(llvm::omp::Clause::OMPC_depend);
}

void OmpStructureChecker::Enter(const parser::OmpIfClause &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_if);

  using dirNameModifier = parser::OmpIfClause::DirectiveNameModifier;
  static std::unordered_map<dirNameModifier, OmpDirectiveSet>
      dirNameModifierMap{{dirNameModifier::Parallel, llvm::omp::parallelSet},
          {dirNameModifier::Target, llvm::omp::targetSet},
          {dirNameModifier::TargetEnterData,
              {llvm::omp::Directive::OMPD_target_enter_data}},
          {dirNameModifier::TargetExitData,
              {llvm::omp::Directive::OMPD_target_exit_data}},
          {dirNameModifier::TargetData,
              {llvm::omp::Directive::OMPD_target_data}},
          {dirNameModifier::TargetUpdate,
              {llvm::omp::Directive::OMPD_target_update}},
          {dirNameModifier::Task, {llvm::omp::Directive::OMPD_task}},
          {dirNameModifier::Taskloop, llvm::omp::taskloopSet}};
  if (const auto &directiveName{
          std::get<std::optional<dirNameModifier>>(x.t)}) {
    auto search{dirNameModifierMap.find(*directiveName)};
    if (search == dirNameModifierMap.end() ||
        !search->second.test(GetContext().directive)) {
      context_
          .Say(GetContext().clauseSource,
              "Unmatched directive name modifier %s on the IF clause"_err_en_US,
              parser::ToUpperCaseLetters(
                  parser::OmpIfClause::EnumToString(*directiveName)))
          .Attach(
              GetContext().directiveSource, "Cannot apply to directive"_en_US);
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpLinearClause &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_linear);

  // 2.7 Loop Construct Restriction
  if ((llvm::omp::doSet | llvm::omp::simdSet).test(GetContext().directive)) {
    if (std::holds_alternative<parser::OmpLinearClause::WithModifier>(x.u)) {
      context_.Say(GetContext().clauseSource,
          "A modifier may not be specified in a LINEAR clause "
          "on the %s directive"_err_en_US,
          ContextDirectiveAsFortran());
    }
  }
}
void OmpStructureChecker::Enter(const parser::OmpMapClause &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_map);
  if (const auto &maptype{std::get<std::optional<parser::OmpMapType>>(x.t)}) {
    using Type = parser::OmpMapType::Type;
    const Type &type{std::get<Type>(maptype->t)};
    switch (GetContext().directive) {
    case llvm::omp::Directive::OMPD_target:
    case llvm::omp::Directive::OMPD_target_teams:
    case llvm::omp::Directive::OMPD_target_teams_distribute:
    case llvm::omp::Directive::OMPD_target_teams_distribute_simd:
    case llvm::omp::Directive::OMPD_target_teams_distribute_parallel_do:
    case llvm::omp::Directive::OMPD_target_teams_distribute_parallel_do_simd:
    case llvm::omp::Directive::OMPD_target_data: {
      if (type != Type::To && type != Type::From && type != Type::Tofrom &&
          type != Type::Alloc) {
        context_.Say(GetContext().clauseSource,
            "Only the TO, FROM, TOFROM, or ALLOC map types are permitted "
            "for MAP clauses on the %s directive"_err_en_US,
            ContextDirectiveAsFortran());
      }
    } break;
    case llvm::omp::Directive::OMPD_target_enter_data: {
      if (type != Type::To && type != Type::Alloc) {
        context_.Say(GetContext().clauseSource,
            "Only the TO or ALLOC map types are permitted "
            "for MAP clauses on the %s directive"_err_en_US,
            ContextDirectiveAsFortran());
      }
    } break;
    case llvm::omp::Directive::OMPD_target_exit_data: {
      if (type != Type::Delete && type != Type::Release && type != Type::From) {
        context_.Say(GetContext().clauseSource,
            "Only the FROM, RELEASE, or DELETE map types are permitted "
            "for MAP clauses on the %s directive"_err_en_US,
            ContextDirectiveAsFortran());
      }
    } break;
    default:
      break;
    }
  }
}
void OmpStructureChecker::Enter(const parser::OmpProcBindClause &) {
  CheckAllowed(llvm::omp::Clause::OMPC_proc_bind);
}
void OmpStructureChecker::Enter(const parser::OmpReductionClause &) {
  CheckAllowed(llvm::omp::Clause::OMPC_reduction);
}

bool OmpStructureChecker::ScheduleModifierHasType(
    const parser::OmpScheduleClause &x,
    const parser::OmpScheduleModifierType::ModType &type) {
  const auto &modifier{
      std::get<std::optional<parser::OmpScheduleModifier>>(x.t)};
  if (modifier) {
    const auto &modType1{
        std::get<parser::OmpScheduleModifier::Modifier1>(modifier->t)};
    const auto &modType2{
        std::get<std::optional<parser::OmpScheduleModifier::Modifier2>>(
            modifier->t)};
    if (modType1.v.v == type || (modType2 && modType2->v.v == type)) {
      return true;
    }
  }
  return false;
}
void OmpStructureChecker::Enter(const parser::OmpScheduleClause &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_schedule);

  // 2.7 Loop Construct Restriction
  if (llvm::omp::doSet.test(GetContext().directive)) {
    const auto &kind{std::get<1>(x.t)};
    const auto &chunk{std::get<2>(x.t)};
    if (chunk) {
      if (kind == parser::OmpScheduleClause::ScheduleType::Runtime ||
          kind == parser::OmpScheduleClause::ScheduleType::Auto) {
        context_.Say(GetContext().clauseSource,
            "When SCHEDULE clause has %s specified, "
            "it must not have chunk size specified"_err_en_US,
            parser::ToUpperCaseLetters(
                parser::OmpScheduleClause::EnumToString(kind)));
      }
    }

    if (ScheduleModifierHasType(
            x, parser::OmpScheduleModifierType::ModType::Nonmonotonic)) {
      if (kind != parser::OmpScheduleClause::ScheduleType::Dynamic &&
          kind != parser::OmpScheduleClause::ScheduleType::Guided) {
        context_.Say(GetContext().clauseSource,
            "The NONMONOTONIC modifier can only be specified with "
            "SCHEDULE(DYNAMIC) or SCHEDULE(GUIDED)"_err_en_US);
      }
    }
  }
}
} // namespace Fortran::semantics
