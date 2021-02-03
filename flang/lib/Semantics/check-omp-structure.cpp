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
#include <algorithm>

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

// 'OmpWorkshareBlockChecker' is used to check the validity of the assignment
// statements and the expressions enclosed in an OpenMP Workshare construct
class OmpWorkshareBlockChecker {
public:
  OmpWorkshareBlockChecker(SemanticsContext &context, parser::CharBlock source)
      : context_{context}, source_{source} {}

  template <typename T> bool Pre(const T &) { return true; }
  template <typename T> void Post(const T &) {}

  bool Pre(const parser::AssignmentStmt &assignment) {
    const auto &var{std::get<parser::Variable>(assignment.t)};
    const auto &expr{std::get<parser::Expr>(assignment.t)};
    const auto *lhs{GetExpr(var)};
    const auto *rhs{GetExpr(expr)};
    Tristate isDefined{semantics::IsDefinedAssignment(
        lhs->GetType(), lhs->Rank(), rhs->GetType(), rhs->Rank())};
    if (isDefined == Tristate::Yes) {
      context_.Say(expr.source,
          "Defined assignment statement is not "
          "allowed in a WORKSHARE construct"_err_en_US);
    }
    return true;
  }

  bool Pre(const parser::Expr &expr) {
    if (const auto *e{GetExpr(expr)}) {
      for (const Symbol &symbol : evaluate::CollectSymbols(*e)) {
        const Symbol &root{GetAssociationRoot(symbol)};
        if (IsFunction(root) &&
            !(root.attrs().test(Attr::ELEMENTAL) ||
                root.attrs().test(Attr::INTRINSIC))) {
          context_.Say(expr.source,
              "User defined non-ELEMENTAL function "
              "'%s' is not allowed in a WORKSHARE construct"_err_en_US,
              root.name());
        }
      }
    }
    return false;
  }

private:
  SemanticsContext &context_;
  parser::CharBlock source_;
};

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
  SetLoopInfo(x);

  if (const auto &doConstruct{
          std::get<std::optional<parser::DoConstruct>>(x.t)}) {
    const auto &doBlock{std::get<parser::Block>(doConstruct->t)};
    CheckNoBranching(doBlock, beginDir.v, beginDir.source);
  }
}
const parser::Name OmpStructureChecker::GetLoopIndex(
    const parser::DoConstruct *x) {
  using Bounds = parser::LoopControl::Bounds;
  return std::get<Bounds>(x->GetLoopControl()->u).name.thing;
}
void OmpStructureChecker::SetLoopInfo(const parser::OpenMPLoopConstruct &x) {
  if (const auto &loopConstruct{
          std::get<std::optional<parser::DoConstruct>>(x.t)}) {
    const parser::DoConstruct *loop{&*loopConstruct};
    if (loop && loop->IsDoNormal()) {
      const parser::Name &itrVal{GetLoopIndex(loop)};
      SetLoopIv(itrVal.symbol);
    }
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

  // TODO: This check needs to be extended while implementing nesting of regions
  // checks.
  if (beginDir.v == llvm::omp::Directive::OMPD_single) {
    HasInvalidWorksharingNesting(
        beginDir.source, {llvm::omp::Directive::OMPD_do});
  }

  PushContextAndClauseSets(beginDir.source, beginDir.v);
  CheckNoBranching(block, beginDir.v, beginDir.source);

  switch (beginDir.v) {
  case llvm::omp::OMPD_workshare:
  case llvm::omp::OMPD_parallel_workshare:
    CheckWorkshareBlockStmts(block, beginDir.source);
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
  const auto &sectionBlocks{std::get<parser::OmpSectionBlocks>(x.t)};
  for (const auto &block : sectionBlocks.v) {
    CheckNoBranching(block, beginDir.v, beginDir.source);
  }
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
    PushContextAndClauseSets(
        dir.source, llvm::omp::Directive::OMPD_end_sections);
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

void OmpStructureChecker::Enter(const parser::OpenMPDeclarativeAllocate &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContextAndClauseSets(dir.source, llvm::omp::Directive::OMPD_allocate);
}

void OmpStructureChecker::Leave(const parser::OpenMPDeclarativeAllocate &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPDeclareTargetConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContext(dir.source, llvm::omp::Directive::OMPD_declare_target);
  const auto &spec{std::get<parser::OmpDeclareTargetSpecifier>(x.t)};
  if (std::holds_alternative<parser::OmpDeclareTargetWithClause>(spec.u)) {
    SetClauseSets(llvm::omp::Directive::OMPD_declare_target);
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPDeclareTargetConstruct &) {
  dirContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPExecutableAllocate &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContextAndClauseSets(dir.source, llvm::omp::Directive::OMPD_allocate);
}

void OmpStructureChecker::Leave(const parser::OpenMPExecutableAllocate &) {
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

void OmpStructureChecker::Leave(const parser::OpenMPFlushConstruct &x) {
  if (FindClause(llvm::omp::Clause::OMPC_acquire) ||
      FindClause(llvm::omp::Clause::OMPC_release) ||
      FindClause(llvm::omp::Clause::OMPC_acq_rel)) {
    if (const auto &flushList{
            std::get<std::optional<parser::OmpObjectList>>(x.t)}) {
      context_.Say(parser::FindSourceLocation(flushList),
          "If memory-order-clause is RELEASE, ACQUIRE, or ACQ_REL, list items "
          "must not be specified on the FLUSH directive"_err_en_US);
    }
  }
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
  const auto &block{std::get<parser::Block>(x.t)};
  CheckNoBranching(block, llvm::omp::Directive::OMPD_critical, dir.source);
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
  case llvm::omp::Directive::OMPD_single:
    PushContextAndClauseSets(dir.source, llvm::omp::Directive::OMPD_end_single);
    break;
  // 2.7.4 end-workshare -> END WORKSHARE [nowait-clause]
  case llvm::omp::Directive::OMPD_workshare:
    PushContextAndClauseSets(
        dir.source, llvm::omp::Directive::OMPD_end_workshare);
    break;
  default:
    // no clauses are allowed
    break;
  }
}

void OmpStructureChecker::Enter(const parser::OpenMPAtomicConstruct &x) {
  std::visit(
      common::visitors{
          [&](const auto &someAtomicConstruct) {
            const auto &dir{std::get<parser::Verbatim>(someAtomicConstruct.t)};
            PushContextAndClauseSets(
                dir.source, llvm::omp::Directive::OMPD_atomic);
          },
      },
      x.u);
}

void OmpStructureChecker::Leave(const parser::OpenMPAtomicConstruct &) {
  dirContext_.pop_back();
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
      const auto &schedClause{std::get<parser::OmpClause::Schedule>(clause->u)};
      if (ScheduleModifierHasType(schedClause.v,
              parser::OmpScheduleModifierType::ModType::Nonmonotonic)) {
        if (FindClause(llvm::omp::Clause::OMPC_ordered)) {
          context_.Say(clause->source,
              "The NONMONOTONIC modifier cannot be specified "
              "if an ORDERED clause is specified"_err_en_US);
        }
        if (ScheduleModifierHasType(schedClause.v,
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
        CheckNotAllowedIfClause(
            llvm::omp::Clause::OMPC_ordered, {llvm::omp::Clause::OMPC_linear});

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
    CheckNotAllowedIfClause(
        llvm::omp::Clause::OMPC_copyprivate, {llvm::omp::Clause::OMPC_nowait});
  }

  CheckRequireAtLeastOneOf();
}

void OmpStructureChecker::Enter(const parser::OmpClause &x) {
  SetContextClause(x);
}

// Following clauses do not have a separate node in parse-tree.h.
CHECK_SIMPLE_CLAUSE(AcqRel, OMPC_acq_rel)
CHECK_SIMPLE_CLAUSE(Acquire, OMPC_acquire)
CHECK_SIMPLE_CLAUSE(AtomicDefaultMemOrder, OMPC_atomic_default_mem_order)
CHECK_SIMPLE_CLAUSE(Affinity, OMPC_affinity)
CHECK_SIMPLE_CLAUSE(Allocate, OMPC_allocate)
CHECK_SIMPLE_CLAUSE(Capture, OMPC_capture)
CHECK_SIMPLE_CLAUSE(Copyin, OMPC_copyin)
CHECK_SIMPLE_CLAUSE(Copyprivate, OMPC_copyprivate)
CHECK_SIMPLE_CLAUSE(Default, OMPC_default)
CHECK_SIMPLE_CLAUSE(Depobj, OMPC_depobj)
CHECK_SIMPLE_CLAUSE(Destroy, OMPC_destroy)
CHECK_SIMPLE_CLAUSE(Detach, OMPC_detach)
CHECK_SIMPLE_CLAUSE(Device, OMPC_device)
CHECK_SIMPLE_CLAUSE(DeviceType, OMPC_device_type)
CHECK_SIMPLE_CLAUSE(DistSchedule, OMPC_dist_schedule)
CHECK_SIMPLE_CLAUSE(DynamicAllocators, OMPC_dynamic_allocators)
CHECK_SIMPLE_CLAUSE(Exclusive, OMPC_exclusive)
CHECK_SIMPLE_CLAUSE(Final, OMPC_final)
CHECK_SIMPLE_CLAUSE(Flush, OMPC_flush)
CHECK_SIMPLE_CLAUSE(From, OMPC_from)
CHECK_SIMPLE_CLAUSE(Hint, OMPC_hint)
CHECK_SIMPLE_CLAUSE(InReduction, OMPC_in_reduction)
CHECK_SIMPLE_CLAUSE(Inclusive, OMPC_inclusive)
CHECK_SIMPLE_CLAUSE(Match, OMPC_match)
CHECK_SIMPLE_CLAUSE(Nontemporal, OMPC_nontemporal)
CHECK_SIMPLE_CLAUSE(Order, OMPC_order)
CHECK_SIMPLE_CLAUSE(Read, OMPC_read)
CHECK_SIMPLE_CLAUSE(ReverseOffload, OMPC_reverse_offload)
CHECK_SIMPLE_CLAUSE(Threadprivate, OMPC_threadprivate)
CHECK_SIMPLE_CLAUSE(Threads, OMPC_threads)
CHECK_SIMPLE_CLAUSE(Inbranch, OMPC_inbranch)
CHECK_SIMPLE_CLAUSE(IsDevicePtr, OMPC_is_device_ptr)
CHECK_SIMPLE_CLAUSE(Lastprivate, OMPC_lastprivate)
CHECK_SIMPLE_CLAUSE(Link, OMPC_link)
CHECK_SIMPLE_CLAUSE(Mergeable, OMPC_mergeable)
CHECK_SIMPLE_CLAUSE(Nogroup, OMPC_nogroup)
CHECK_SIMPLE_CLAUSE(Notinbranch, OMPC_notinbranch)
CHECK_SIMPLE_CLAUSE(Nowait, OMPC_nowait)
CHECK_SIMPLE_CLAUSE(ProcBind, OMPC_proc_bind)
CHECK_SIMPLE_CLAUSE(Reduction, OMPC_reduction)
CHECK_SIMPLE_CLAUSE(Release, OMPC_release)
CHECK_SIMPLE_CLAUSE(Relaxed, OMPC_relaxed)
CHECK_SIMPLE_CLAUSE(SeqCst, OMPC_seq_cst)
CHECK_SIMPLE_CLAUSE(Simd, OMPC_simd)
CHECK_SIMPLE_CLAUSE(TaskReduction, OMPC_task_reduction)
CHECK_SIMPLE_CLAUSE(To, OMPC_to)
CHECK_SIMPLE_CLAUSE(UnifiedAddress, OMPC_unified_address)
CHECK_SIMPLE_CLAUSE(UnifiedSharedMemory, OMPC_unified_shared_memory)
CHECK_SIMPLE_CLAUSE(Uniform, OMPC_uniform)
CHECK_SIMPLE_CLAUSE(Unknown, OMPC_unknown)
CHECK_SIMPLE_CLAUSE(Untied, OMPC_untied)
CHECK_SIMPLE_CLAUSE(UseDevicePtr, OMPC_use_device_ptr)
CHECK_SIMPLE_CLAUSE(UsesAllocators, OMPC_uses_allocators)
CHECK_SIMPLE_CLAUSE(Update, OMPC_update)
CHECK_SIMPLE_CLAUSE(UseDeviceAddr, OMPC_use_device_addr)
CHECK_SIMPLE_CLAUSE(Write, OMPC_write)

CHECK_REQ_SCALAR_INT_CLAUSE(Allocator, OMPC_allocator)
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

void OmpStructureChecker::Enter(const parser::OmpClause::Shared &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_shared);
  CheckIsVarPartOfAnotherVar(x.v);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Private &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_private);
  CheckIsVarPartOfAnotherVar(x.v);
  CheckIntentInPointer(x.v, llvm::omp::Clause::OMPC_private);
}

void OmpStructureChecker::CheckIsVarPartOfAnotherVar(
    const parser::OmpObjectList &objList) {

  for (const auto &ompObject : objList.v) {
    std::visit(
        common::visitors{
            [&](const parser::Designator &designator) {
              if (std::get_if<parser::DataRef>(&designator.u)) {
                if ((parser::Unwrap<parser::StructureComponent>(ompObject)) ||
                    (parser::Unwrap<parser::ArrayElement>(ompObject))) {
                  context_.Say(GetContext().clauseSource,
                      "A variable that is part of another variable (as an "
                      "array or structure element)"
                      " cannot appear in a PRIVATE or SHARED clause."_err_en_US);
                }
              }
            },
            [&](const parser::Name &name) {},
        },
        ompObject.u);
  }
}
void OmpStructureChecker::Enter(const parser::OmpClause::Firstprivate &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_firstprivate);
  CheckIsLoopIvPartOfClause(llvmOmpClause::OMPC_firstprivate, x.v);
}
void OmpStructureChecker::CheckIsLoopIvPartOfClause(
    llvmOmpClause clause, const parser::OmpObjectList &ompObjectList) {
  for (const auto &ompObject : ompObjectList.v) {
    if (const parser::Name * name{parser::Unwrap<parser::Name>(ompObject)}) {
      if (name->symbol == GetContext().loopIV) {
        context_.Say(name->source,
            "DO iteration variable %s is not allowed in %s clause."_err_en_US,
            name->ToString(),
            parser::ToUpperCaseLetters(getClauseName(clause).str()));
      }
    }
  }
}
// Following clauses have a seperate node in parse-tree.h.
// Atomic-clause
CHECK_SIMPLE_PARSER_CLAUSE(OmpAtomicRead, OMPC_read)
CHECK_SIMPLE_PARSER_CLAUSE(OmpAtomicWrite, OMPC_write)
CHECK_SIMPLE_PARSER_CLAUSE(OmpAtomicUpdate, OMPC_update)
CHECK_SIMPLE_PARSER_CLAUSE(OmpAtomicCapture, OMPC_capture)

void OmpStructureChecker::Leave(const parser::OmpAtomicRead &) {
  CheckNotAllowedIfClause(llvm::omp::Clause::OMPC_read,
      {llvm::omp::Clause::OMPC_release, llvm::omp::Clause::OMPC_acq_rel});
}
void OmpStructureChecker::Leave(const parser::OmpAtomicWrite &) {
  CheckNotAllowedIfClause(llvm::omp::Clause::OMPC_write,
      {llvm::omp::Clause::OMPC_acquire, llvm::omp::Clause::OMPC_acq_rel});
}
void OmpStructureChecker::Leave(const parser::OmpAtomicUpdate &) {
  CheckNotAllowedIfClause(llvm::omp::Clause::OMPC_update,
      {llvm::omp::Clause::OMPC_acquire, llvm::omp::Clause::OMPC_acq_rel});
}
// OmpAtomic node represents atomic directive without atomic-clause.
// atomic-clause - READ,WRITE,UPDATE,CAPTURE.
void OmpStructureChecker::Leave(const parser::OmpAtomic &) {
  if (const auto *clause{FindClause(llvm::omp::Clause::OMPC_acquire)}) {
    context_.Say(clause->source,
        "Clause ACQUIRE is not allowed on the ATOMIC directive"_err_en_US);
  }
  if (const auto *clause{FindClause(llvm::omp::Clause::OMPC_acq_rel)}) {
    context_.Say(clause->source,
        "Clause ACQ_REL is not allowed on the ATOMIC directive"_err_en_US);
  }
}
// Restrictions specific to each clause are implemented apart from the
// generalized restrictions.
void OmpStructureChecker::Enter(const parser::OmpClause::Aligned &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_aligned);

  if (const auto &expr{
          std::get<std::optional<parser::ScalarIntConstantExpr>>(x.v.t)}) {
    RequiresConstantPositiveParameter(llvm::omp::Clause::OMPC_aligned, *expr);
  }
  // 2.8.1 TODO: list-item attribute check
}
void OmpStructureChecker::Enter(const parser::OmpClause::Defaultmap &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_defaultmap);
  using VariableCategory = parser::OmpDefaultmapClause::VariableCategory;
  if (!std::get<std::optional<VariableCategory>>(x.v.t)) {
    context_.Say(GetContext().clauseSource,
        "The argument TOFROM:SCALAR must be specified on the DEFAULTMAP "
        "clause"_err_en_US);
  }
}
void OmpStructureChecker::Enter(const parser::OmpClause::If &x) {
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
          std::get<std::optional<dirNameModifier>>(x.v.t)}) {
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

void OmpStructureChecker::Enter(const parser::OmpClause::Linear &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_linear);

  // 2.7 Loop Construct Restriction
  if ((llvm::omp::doSet | llvm::omp::simdSet).test(GetContext().directive)) {
    if (std::holds_alternative<parser::OmpLinearClause::WithModifier>(x.v.u)) {
      context_.Say(GetContext().clauseSource,
          "A modifier may not be specified in a LINEAR clause "
          "on the %s directive"_err_en_US,
          ContextDirectiveAsFortran());
    }
  }
}

void OmpStructureChecker::CheckAllowedMapTypes(
    const parser::OmpMapType::Type &type,
    const std::list<parser::OmpMapType::Type> &allowedMapTypeList) {
  const auto found{std::find(
      std::begin(allowedMapTypeList), std::end(allowedMapTypeList), type)};
  if (found == std::end(allowedMapTypeList)) {
    std::string commaSeperatedMapTypes;
    llvm::interleave(
        allowedMapTypeList.begin(), allowedMapTypeList.end(),
        [&](const parser::OmpMapType::Type &mapType) {
          commaSeperatedMapTypes.append(parser::ToUpperCaseLetters(
              parser::OmpMapType::EnumToString(mapType)));
        },
        [&] { commaSeperatedMapTypes.append(", "); });
    context_.Say(GetContext().clauseSource,
        "Only the %s map types are permitted "
        "for MAP clauses on the %s directive"_err_en_US,
        commaSeperatedMapTypes, ContextDirectiveAsFortran());
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Map &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_map);

  if (const auto &maptype{std::get<std::optional<parser::OmpMapType>>(x.v.t)}) {
    using Type = parser::OmpMapType::Type;
    const Type &type{std::get<Type>(maptype->t)};
    switch (GetContext().directive) {
    case llvm::omp::Directive::OMPD_target:
    case llvm::omp::Directive::OMPD_target_teams:
    case llvm::omp::Directive::OMPD_target_teams_distribute:
    case llvm::omp::Directive::OMPD_target_teams_distribute_simd:
    case llvm::omp::Directive::OMPD_target_teams_distribute_parallel_do:
    case llvm::omp::Directive::OMPD_target_teams_distribute_parallel_do_simd:
    case llvm::omp::Directive::OMPD_target_data:
      CheckAllowedMapTypes(
          type, {Type::To, Type::From, Type::Tofrom, Type::Alloc});
      break;
    case llvm::omp::Directive::OMPD_target_enter_data:
      CheckAllowedMapTypes(type, {Type::To, Type::Alloc});
      break;
    case llvm::omp::Directive::OMPD_target_exit_data:
      CheckAllowedMapTypes(type, {Type::From, Type::Release, Type::Delete});
      break;
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
void OmpStructureChecker::Enter(const parser::OmpClause::Schedule &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_schedule);
  const parser::OmpScheduleClause &scheduleClause = x.v;

  // 2.7 Loop Construct Restriction
  if (llvm::omp::doSet.test(GetContext().directive)) {
    const auto &kind{std::get<1>(scheduleClause.t)};
    const auto &chunk{std::get<2>(scheduleClause.t)};
    if (chunk) {
      if (kind == parser::OmpScheduleClause::ScheduleType::Runtime ||
          kind == parser::OmpScheduleClause::ScheduleType::Auto) {
        context_.Say(GetContext().clauseSource,
            "When SCHEDULE clause has %s specified, "
            "it must not have chunk size specified"_err_en_US,
            parser::ToUpperCaseLetters(
                parser::OmpScheduleClause::EnumToString(kind)));
      }
      if (const auto &chunkExpr{std::get<std::optional<parser::ScalarIntExpr>>(
              scheduleClause.t)}) {
        RequiresPositiveParameter(
            llvm::omp::Clause::OMPC_schedule, *chunkExpr, "chunk size");
      }
    }

    if (ScheduleModifierHasType(scheduleClause,
            parser::OmpScheduleModifierType::ModType::Nonmonotonic)) {
      if (kind != parser::OmpScheduleClause::ScheduleType::Dynamic &&
          kind != parser::OmpScheduleClause::ScheduleType::Guided) {
        context_.Say(GetContext().clauseSource,
            "The NONMONOTONIC modifier can only be specified with "
            "SCHEDULE(DYNAMIC) or SCHEDULE(GUIDED)"_err_en_US);
      }
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Depend &x) {
  CheckAllowed(llvm::omp::Clause::OMPC_depend);
  if (const auto *inOut{std::get_if<parser::OmpDependClause::InOut>(&x.v.u)}) {
    const auto &designators{std::get<std::list<parser::Designator>>(inOut->t)};
    for (const auto &ele : designators) {
      if (const auto *dataRef{std::get_if<parser::DataRef>(&ele.u)}) {
        CheckDependList(*dataRef);
        if (const auto *arr{
                std::get_if<common::Indirection<parser::ArrayElement>>(
                    &dataRef->u)}) {
          CheckDependArraySection(*arr, GetLastName(*dataRef));
        }
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

void OmpStructureChecker::CheckDependList(const parser::DataRef &d) {
  std::visit(
      common::visitors{
          [&](const common::Indirection<parser::ArrayElement> &elem) {
            // Check if the base element is valid on Depend Clause
            CheckDependList(elem.value().base);
          },
          [&](const common::Indirection<parser::StructureComponent> &) {
            context_.Say(GetContext().clauseSource,
                "A variable that is part of another variable "
                "(such as an element of a structure) but is not an array "
                "element or an array section cannot appear in a DEPEND "
                "clause"_err_en_US);
          },
          [&](const common::Indirection<parser::CoindexedNamedObject> &) {
            context_.Say(GetContext().clauseSource,
                "Coarrays are not supported in DEPEND clause"_err_en_US);
          },
          [&](const parser::Name &) { return; },
      },
      d.u);
}

void OmpStructureChecker::CheckDependArraySection(
    const common::Indirection<parser::ArrayElement> &arr,
    const parser::Name &name) {
  for (const auto &subscript : arr.value().subscripts) {
    if (const auto *triplet{
            std::get_if<parser::SubscriptTriplet>(&subscript.u)}) {
      if (std::get<2>(triplet->t)) {
        context_.Say(GetContext().clauseSource,
            "Stride should not be specified for array section in DEPEND "
            "clause"_err_en_US);
      }
      const auto &lower{std::get<0>(triplet->t)};
      const auto &upper{std::get<1>(triplet->t)};
      if (lower && upper) {
        const auto lval{GetIntValue(lower)};
        const auto uval{GetIntValue(upper)};
        if (lval && uval && *uval < *lval) {
          context_.Say(GetContext().clauseSource,
              "'%s' in DEPEND clause is a zero size array section"_err_en_US,
              name.ToString());
          break;
        }
      }
    }
  }
}

void OmpStructureChecker::CheckIntentInPointer(
    const parser::OmpObjectList &objectList, const llvm::omp::Clause clause) {
  std::vector<const Symbol *> symbols;
  GetSymbolsInObjectList(objectList, symbols);
  for (const auto *symbol : symbols) {
    if (IsPointer(*symbol) && IsIntentIn(*symbol)) {
      context_.Say(GetContext().clauseSource,
          "Pointer '%s' with the INTENT(IN) attribute may not appear "
          "in a %s clause"_err_en_US,
          symbol->name(),
          parser::ToUpperCaseLetters(getClauseName(clause).str()));
    }
  }
}

void OmpStructureChecker::GetSymbolsInObjectList(
    const parser::OmpObjectList &objectList,
    std::vector<const Symbol *> &symbols) {
  for (const auto &ompObject : objectList.v) {
    if (const auto *name{parser::Unwrap<parser::Name>(ompObject)}) {
      if (const auto *symbol{name->symbol}) {
        if (const auto *commonBlockDetails{
                symbol->detailsIf<CommonBlockDetails>()}) {
          for (const auto &object : commonBlockDetails->objects()) {
            symbols.emplace_back(&object->GetUltimate());
          }
        } else {
          symbols.emplace_back(&symbol->GetUltimate());
        }
      }
    }
  }
}

void OmpStructureChecker::CheckWorkshareBlockStmts(
    const parser::Block &block, parser::CharBlock source) {
  OmpWorkshareBlockChecker ompWorkshareBlockChecker{context_, source};

  for (auto it{block.begin()}; it != block.end(); ++it) {
    if (parser::Unwrap<parser::AssignmentStmt>(*it) ||
        parser::Unwrap<parser::ForallStmt>(*it) ||
        parser::Unwrap<parser::ForallConstruct>(*it) ||
        parser::Unwrap<parser::WhereStmt>(*it) ||
        parser::Unwrap<parser::WhereConstruct>(*it)) {
      parser::Walk(*it, ompWorkshareBlockChecker);
    } else if (const auto *ompConstruct{
                   parser::Unwrap<parser::OpenMPConstruct>(*it)}) {
      if (const auto *ompAtomicConstruct{
              std::get_if<parser::OpenMPAtomicConstruct>(&ompConstruct->u)}) {
        // Check if assignment statements in the enclosing OpenMP Atomic
        // construct are allowed in the Workshare construct
        parser::Walk(*ompAtomicConstruct, ompWorkshareBlockChecker);
      } else if (const auto *ompCriticalConstruct{
                     std::get_if<parser::OpenMPCriticalConstruct>(
                         &ompConstruct->u)}) {
        // All the restrictions on the Workshare construct apply to the
        // statements in the enclosing critical constructs
        const auto &criticalBlock{
            std::get<parser::Block>(ompCriticalConstruct->t)};
        CheckWorkshareBlockStmts(criticalBlock, source);
      } else {
        // Check if OpenMP constructs enclosed in the Workshare construct are
        // 'Parallel' constructs
        auto currentDir{llvm::omp::Directive::OMPD_unknown};
        const OmpDirectiveSet parallelDirSet{
            llvm::omp::Directive::OMPD_parallel,
            llvm::omp::Directive::OMPD_parallel_do,
            llvm::omp::Directive::OMPD_parallel_sections,
            llvm::omp::Directive::OMPD_parallel_workshare,
            llvm::omp::Directive::OMPD_parallel_do_simd};

        if (const auto *ompBlockConstruct{
                std::get_if<parser::OpenMPBlockConstruct>(&ompConstruct->u)}) {
          const auto &beginBlockDir{
              std::get<parser::OmpBeginBlockDirective>(ompBlockConstruct->t)};
          const auto &beginDir{
              std::get<parser::OmpBlockDirective>(beginBlockDir.t)};
          currentDir = beginDir.v;
        } else if (const auto *ompLoopConstruct{
                       std::get_if<parser::OpenMPLoopConstruct>(
                           &ompConstruct->u)}) {
          const auto &beginLoopDir{
              std::get<parser::OmpBeginLoopDirective>(ompLoopConstruct->t)};
          const auto &beginDir{
              std::get<parser::OmpLoopDirective>(beginLoopDir.t)};
          currentDir = beginDir.v;
        } else if (const auto *ompSectionsConstruct{
                       std::get_if<parser::OpenMPSectionsConstruct>(
                           &ompConstruct->u)}) {
          const auto &beginSectionsDir{
              std::get<parser::OmpBeginSectionsDirective>(
                  ompSectionsConstruct->t)};
          const auto &beginDir{
              std::get<parser::OmpSectionsDirective>(beginSectionsDir.t)};
          currentDir = beginDir.v;
        }

        if (!parallelDirSet.test(currentDir)) {
          context_.Say(source,
              "OpenMP constructs enclosed in WORKSHARE construct may consist "
              "of ATOMIC, CRITICAL or PARALLEL constructs only"_err_en_US);
        }
      }
    } else {
      context_.Say(source,
          "The structured block in a WORKSHARE construct may consist of only "
          "SCALAR or ARRAY assignments, FORALL or WHERE statements, "
          "FORALL, WHERE, ATOMIC, CRITICAL or PARALLEL constructs"_err_en_US);
    }
  }
}

} // namespace Fortran::semantics
