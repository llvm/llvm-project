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

namespace Fortran::semantics {

// Use when clause falls under 'struct OmpClause' in 'parse-tree.h'.
#define CHECK_SIMPLE_CLAUSE(X, Y) \
  void OmpStructureChecker::Enter(const parser::OmpClause::X &) { \
    CheckAllowed(llvm::omp::Clause::Y); \
  }

#define CHECK_REQ_CONSTANT_SCALAR_INT_CLAUSE(X, Y) \
  void OmpStructureChecker::Enter(const parser::OmpClause::X &c) { \
    CheckAllowed(llvm::omp::Clause::Y); \
    RequiresConstantPositiveParameter(llvm::omp::Clause::Y, c.v); \
  }

#define CHECK_REQ_SCALAR_INT_CLAUSE(X, Y) \
  void OmpStructureChecker::Enter(const parser::OmpClause::X &c) { \
    CheckAllowed(llvm::omp::Clause::Y); \
    RequiresPositiveParameter(llvm::omp::Clause::Y, c.v); \
  }

// Use when clause don't falls under 'struct OmpClause' in 'parse-tree.h'.
#define CHECK_SIMPLE_PARSER_CLAUSE(X, Y) \
  void OmpStructureChecker::Enter(const parser::X &) { \
    CheckAllowed(llvm::omp::Y); \
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

void OmpStructureChecker::Enter(const parser::OpenMPConstruct &) {
  // 2.8.1 TODO: Simd Construct with Ordered Construct Nesting check
}

void OmpStructureChecker::Enter(const parser::OpenMPLoopConstruct &x) {
  const auto &beginLoopDir{std::get<parser::OmpBeginLoopDirective>(x.t)};
  const auto &beginDir{std::get<parser::OmpLoopDirective>(beginLoopDir.t)};

  // check matching, End directive is optional
  if (const auto &endLoopDir{
          std::get<std::optional<parser::OmpEndLoopDirective>>(x.t)}) {
    const auto &endDir{
        std::get<parser::OmpLoopDirective>(endLoopDir.value().t)};

    CheckMatching<parser::OmpLoopDirective>(beginDir, endDir);
  }

  if (beginDir.v != llvm::omp::Directive::OMPD_do) {
    PushContextAndClauseSets(beginDir.source, beginDir.v);
  } else {
    // 2.7.1 do-clause -> private-clause |
    //                    firstprivate-clause |
    //                    lastprivate-clause |
    //                    linear-clause |
    //                    reduction-clause |
    //                    schedule-clause |
    //                    collapse-clause |
    //                    ordered-clause

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
    PushContextAndClauseSets(beginDir.source, llvm::omp::Directive::OMPD_do);
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPLoopConstruct &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OmpEndLoopDirective &x) {
  const auto &dir{std::get<parser::OmpLoopDirective>(x.t)};
  ResetPartialContext(dir.source);
  switch (dir.v) {
  // 2.7.1 end-do -> END DO [nowait-clause]
  // 2.8.3 end-do-simd -> END DO SIMD [nowait-clause]
  case llvm::omp::Directive::OMPD_do:
  case llvm::omp::Directive::OMPD_do_simd:
    SetClauseSets(dir.v);
    break;
  default:
    // no clauses are allowed
    break;
  }
}

void OmpStructureChecker::Enter(const parser::OpenMPBlockConstruct &x) {
  const auto &beginBlockDir{std::get<parser::OmpBeginBlockDirective>(x.t)};
  const auto &endBlockDir{std::get<parser::OmpEndBlockDirective>(x.t)};
  const auto &beginDir{std::get<parser::OmpBlockDirective>(beginBlockDir.t)};
  const auto &endDir{std::get<parser::OmpBlockDirective>(endBlockDir.t)};
  const parser::Block &block{std::get<parser::Block>(x.t)};

  CheckMatching<parser::OmpBlockDirective>(beginDir, endDir);

  PushContextAndClauseSets(beginDir.source, beginDir.v);

  switch (beginDir.v) {
  case llvm::omp::OMPD_parallel:
    CheckNoBranching(block, llvm::omp::OMPD_parallel, beginDir.source);
    break;
  default:
    break;
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPBlockConstruct &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPSectionsConstruct &x) {
  const auto &beginSectionsDir{
      std::get<parser::OmpBeginSectionsDirective>(x.t)};
  const auto &endSectionsDir{std::get<parser::OmpEndSectionsDirective>(x.t)};
  const auto &beginDir{
      std::get<parser::OmpSectionsDirective>(beginSectionsDir.t)};
  const auto &endDir{std::get<parser::OmpSectionsDirective>(endSectionsDir.t)};
  CheckMatching<parser::OmpSectionsDirective>(beginDir, endDir);

  PushContextAndClauseSets(beginDir.source, beginDir.v);
}

void OmpStructureChecker::Leave(const parser::OpenMPSectionsConstruct &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OmpEndSectionsDirective &x) {
  const auto &dir{std::get<parser::OmpSectionsDirective>(x.t)};
  ResetPartialContext(dir.source);
  switch (dir.v) {
    // 2.7.2 end-sections -> END SECTIONS [nowait-clause]
  case llvm::omp::Directive::OMPD_sections:
    SetContextDirectiveEnum(llvm::omp::Directive::OMPD_end_sections);
    SetContextAllowedOnce(OmpClauseSet{llvm::omp::Clause::OMPC_nowait});
    break;
  default:
    // no clauses are allowed
    break;
  }
}

void OmpStructureChecker::Enter(const parser::OpenMPDeclareSimdConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContextAndClauseSets(dir.source, llvm::omp::Directive::OMPD_declare_simd);
}

void OmpStructureChecker::Leave(const parser::OpenMPDeclareSimdConstruct &) {
  dirContext_.pop_back();
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
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(
    const parser::OpenMPSimpleStandaloneConstruct &x) {
  const auto &dir{std::get<parser::OmpSimpleStandaloneDirective>(x.t)};
  PushContextAndClauseSets(dir.source, dir.v);
}

void OmpStructureChecker::Leave(
    const parser::OpenMPSimpleStandaloneConstruct &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPFlushConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContextAndClauseSets(dir.source, llvm::omp::Directive::OMPD_flush);
}

void OmpStructureChecker::Leave(const parser::OpenMPFlushConstruct &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPCancelConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContextAndClauseSets(dir.source, llvm::omp::Directive::OMPD_cancel);
}

void OmpStructureChecker::Leave(const parser::OpenMPCancelConstruct &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPCriticalConstruct &x) {
  const auto &dir{std::get<parser::OmpCriticalDirective>(x.t)};
  PushContextAndClauseSets(dir.source, llvm::omp::Directive::OMPD_critical);
}

void OmpStructureChecker::Leave(const parser::OpenMPCriticalConstruct &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(
    const parser::OpenMPCancellationPointConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContextAndClauseSets(
      dir.source, llvm::omp::Directive::OMPD_cancellation_point);
}

void OmpStructureChecker::Leave(
    const parser::OpenMPCancellationPointConstruct &) {
  dirContext_.pop_back();
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

// Clauses
// Mainly categorized as
// 1. Checks on 'OmpClauseList' from 'parse-tree.h'.
// 2. Checks on clauses which fall under 'struct OmpClause' from parse-tree.h.
// 3. Checks on clauses which are not in 'struct OmpClause' from parse-tree.h.

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

// Following clauses do not have a seperate node in parse-tree.h.
// They fall under 'struct OmpClause' in parse-tree.h.
CHECK_SIMPLE_CLAUSE(Copyin, OMPC_copyin)
CHECK_SIMPLE_CLAUSE(Copyprivate, OMPC_copyprivate)
CHECK_SIMPLE_CLAUSE(Device, OMPC_device)
CHECK_SIMPLE_CLAUSE(Final, OMPC_final)
CHECK_SIMPLE_CLAUSE(Firstprivate, OMPC_firstprivate)
CHECK_SIMPLE_CLAUSE(From, OMPC_from)
CHECK_SIMPLE_CLAUSE(Inbranch, OMPC_inbranch)
CHECK_SIMPLE_CLAUSE(IsDevicePtr, OMPC_is_device_ptr)
CHECK_SIMPLE_CLAUSE(Lastprivate, OMPC_lastprivate)
CHECK_SIMPLE_CLAUSE(Link, OMPC_link)
CHECK_SIMPLE_CLAUSE(Mergeable, OMPC_mergeable)
CHECK_SIMPLE_CLAUSE(Nogroup, OMPC_nogroup)
CHECK_SIMPLE_CLAUSE(Notinbranch, OMPC_notinbranch)
CHECK_SIMPLE_CLAUSE(Private, OMPC_private)
CHECK_SIMPLE_CLAUSE(Shared, OMPC_shared)
CHECK_SIMPLE_CLAUSE(To, OMPC_to)
CHECK_SIMPLE_CLAUSE(Uniform, OMPC_uniform)
CHECK_SIMPLE_CLAUSE(Untied, OMPC_untied)
CHECK_SIMPLE_CLAUSE(UseDevicePtr, OMPC_use_device_ptr)

CHECK_REQ_SCALAR_INT_CLAUSE(Grainsize, OMPC_grainsize)
CHECK_REQ_SCALAR_INT_CLAUSE(NumTasks, OMPC_num_tasks)
CHECK_REQ_SCALAR_INT_CLAUSE(NumTeams, OMPC_num_teams)
CHECK_REQ_SCALAR_INT_CLAUSE(NumThreads, OMPC_num_threads)
CHECK_REQ_SCALAR_INT_CLAUSE(Priority, OMPC_priority)
CHECK_REQ_SCALAR_INT_CLAUSE(ThreadLimit, OMPC_thread_limit)

CHECK_REQ_CONSTANT_SCALAR_INT_CLAUSE(Collapse, OMPC_collapse)
CHECK_REQ_CONSTANT_SCALAR_INT_CLAUSE(Safelen, OMPC_safelen)
CHECK_REQ_CONSTANT_SCALAR_INT_CLAUSE(Simdlen, OMPC_simdlen)

// Restrictions specific to each clause are implemented apart from the
// generalized restrictions.
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

// Following clauses have a seperate node in parse-tree.h.
CHECK_SIMPLE_PARSER_CLAUSE(OmpAllocateClause, OMPC_allocate)
CHECK_SIMPLE_PARSER_CLAUSE(OmpDefaultClause, OMPC_default)
CHECK_SIMPLE_PARSER_CLAUSE(OmpDependClause, OMPC_depend)
CHECK_SIMPLE_PARSER_CLAUSE(OmpDistScheduleClause, OMPC_dist_schedule)
CHECK_SIMPLE_PARSER_CLAUSE(OmpNowait, OMPC_nowait)
CHECK_SIMPLE_PARSER_CLAUSE(OmpProcBindClause, OMPC_proc_bind)
CHECK_SIMPLE_PARSER_CLAUSE(OmpReductionClause, OMPC_reduction)

// Restrictions specific to each clause are implemented apart from the
// generalized restrictions.
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
void OmpStructureChecker::Enter(const parser::OmpDefaultmapClause &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_defaultmap);
  using VariableCategory = parser::OmpDefaultmapClause::VariableCategory;
  if (!std::get<std::optional<VariableCategory>>(x.t)) {
    context_.Say(GetContext().clauseSource,
        "The argument TOFROM:SCALAR must be specified on the DEFAULTMAP "
        "clause"_err_en_US);
  }
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
      if (const auto &chunkExpr{
              std::get<std::optional<parser::ScalarIntExpr>>(x.t)}) {
        RequiresPositiveParameter(
            llvm::omp::Clause::OMPC_schedule, *chunkExpr, "chunk size");
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

llvm::StringRef OmpStructureChecker::getClauseName(llvm::omp::Clause clause) {
  return llvm::omp::getOpenMPClauseName(clause);
}

llvm::StringRef OmpStructureChecker::getDirectiveName(
    llvm::omp::Directive directive) {
  return llvm::omp::getOpenMPDirectiveName(directive);
}

} // namespace Fortran::semantics
