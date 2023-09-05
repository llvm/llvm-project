//===-- lib/Parser/openmp-parsers.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Top-level grammar specification for OpenMP.
// See OpenMP-4.5-grammar.txt for documentation.

#include "basic-parsers.h"
#include "expr-parsers.h"
#include "misc-parsers.h"
#include "stmt-parser.h"
#include "token-parsers.h"
#include "type-parser-implementation.h"
#include "flang/Parser/parse-tree.h"

// OpenMP Directives and Clauses
namespace Fortran::parser {

constexpr auto startOmpLine = skipStuffBeforeStatement >> "!$OMP "_sptok;
constexpr auto endOmpLine = space >> endOfLine;

// OpenMP Clauses
// 2.15.3.1 DEFAULT (PRIVATE | FIRSTPRIVATE | SHARED | NONE)
TYPE_PARSER(construct<OmpDefaultClause>(
    "PRIVATE" >> pure(common::OmpDefaultClauseKind::Private) ||
    "FIRSTPRIVATE" >> pure(common::OmpDefaultClauseKind::Firstprivate) ||
    "SHARED" >> pure(common::OmpDefaultClauseKind::Shared) ||
    "NONE" >> pure(common::OmpDefaultClauseKind::None)))

// 2.5 PROC_BIND (MASTER | CLOSE | SPREAD)
TYPE_PARSER(construct<OmpProcBindClause>(
    "CLOSE" >> pure(common::OmpProcBindClauseKind::Close) ||
    "MASTER" >> pure(common::OmpProcBindClauseKind::Master) ||
    "SPREAD" >> pure(common::OmpProcBindClauseKind::Spread)))

// 2.15.5.1 MAP ([ [ALWAYS[,]] map-type : ] variable-name-list)
//          map-type -> TO | FROM | TOFROM | ALLOC | RELEASE | DELETE
TYPE_PARSER(construct<OmpMapType>(
    maybe("ALWAYS" >> construct<OmpMapType::Always>() / maybe(","_tok)),
    ("TO"_id >> pure(common::OmpMapKind::To) ||
        "FROM" >> pure(common::OmpMapKind::From) ||
        "TOFROM" >> pure(common::OmpMapKind::Tofrom) ||
        "ALLOC" >> pure(common::OmpMapKind::Alloc) ||
        "RELEASE" >> pure(common::OmpMapKind::Release) ||
        "DELETE" >> pure(common::OmpMapKind::Delete)) /
        ":"))

TYPE_PARSER(construct<OmpMapClause>(
    maybe(Parser<OmpMapType>{}), Parser<OmpObjectList>{}))

// [OpenMP 5.0]
// 2.19.7.2 defaultmap(implicit-behavior[:variable-category])
//  implicit-behavior -> ALLOC | TO | FROM | TOFROM | FIRSRTPRIVATE | NONE |
//  DEFAULT
//  variable-category -> SCALAR | AGGREGATE | ALLOCATABLE | POINTER
TYPE_PARSER(construct<OmpDefaultmapClause>(
    construct<common::OmpDefaultmapClauseImplicitBehavior>(
        "ALLOC" >> pure(common::OmpDefaultmapClauseImplicitBehavior::Alloc) ||
        "TO"_id >> pure(common::OmpDefaultmapClauseImplicitBehavior::To) ||
        "FROM" >> pure(common::OmpDefaultmapClauseImplicitBehavior::From) ||
        "TOFROM" >> pure(common::OmpDefaultmapClauseImplicitBehavior::Tofrom) ||
        "FIRSTPRIVATE" >>
            pure(common::OmpDefaultmapClauseImplicitBehavior::Firstprivate) ||
        "NONE" >> pure(common::OmpDefaultmapClauseImplicitBehavior::None) ||
        "DEFAULT" >>
            pure(common::OmpDefaultmapClauseImplicitBehavior::Default)),
    maybe(":" >>
        construct<common::OmpDefaultmapClauseVariableCategory>("SCALAR" >>
                pure(common::OmpDefaultmapClauseVariableCategory::Scalar) ||
            "AGGREGATE" >>
                pure(common::OmpDefaultmapClauseVariableCategory::Aggregate) ||
            "ALLOCATABLE" >>
                pure(
                    common::OmpDefaultmapClauseVariableCategory::Allocatable) ||
            "POINTER" >>
                pure(common::OmpDefaultmapClauseVariableCategory::Pointer)))))

// 2.7.1 SCHEDULE ([modifier1 [, modifier2]:]kind[, chunk_size])
//       Modifier ->  MONITONIC | NONMONOTONIC | SIMD
//       kind -> STATIC | DYNAMIC | GUIDED | AUTO | RUNTIME
//       chunk_size -> ScalarIntExpr
TYPE_PARSER(construct<OmpScheduleModifierType>(
    "MONOTONIC" >> pure(common::OmpScheduleModifierKind::Monotonic) ||
    "NONMONOTONIC" >> pure(common::OmpScheduleModifierKind::Nonmonotonic) ||
    "SIMD" >> pure(common::OmpScheduleModifierKind::Simd)))

TYPE_PARSER(construct<OmpScheduleModifier>(Parser<OmpScheduleModifierType>{},
    maybe("," >> Parser<OmpScheduleModifierType>{}) / ":"))

TYPE_PARSER(construct<OmpScheduleClause>(maybe(Parser<OmpScheduleModifier>{}),
    "STATIC" >> pure(common::OmpScheduleClauseKind::Static) ||
        "DYNAMIC" >> pure(common::OmpScheduleClauseKind::Dynamic) ||
        "GUIDED" >> pure(common::OmpScheduleClauseKind::Guided) ||
        "AUTO" >> pure(common::OmpScheduleClauseKind::Auto) ||
        "RUNTIME" >> pure(common::OmpScheduleClauseKind::Runtime),
    maybe("," >> scalarIntExpr)))

// device([ device-modifier :] scalar-integer-expression)
TYPE_PARSER(construct<OmpDeviceClause>(
    maybe(
        ("ANCESTOR" >> pure(common::OmpDeviceClauseDeviceModifier::Ancestor) ||
            "DEVICE_NUM" >>
                pure(common::OmpDeviceClauseDeviceModifier::Device_Num)) /
        ":"),
    scalarIntExpr))

// device_type(any | host | nohost)
TYPE_PARSER(construct<OmpDeviceTypeClause>(
    "ANY" >> pure(common::OmpDeviceTypeClauseKind::Any) ||
    "HOST" >> pure(common::OmpDeviceTypeClauseKind::Host) ||
    "NOHOST" >> pure(common::OmpDeviceTypeClauseKind::Nohost)))

// 2.12 IF (directive-name-modifier: scalar-logical-expr)
TYPE_PARSER(construct<OmpIfClause>(
    maybe(
        ("PARALLEL" >>
                pure(common::OmpIfClauseDirectiveNameModifier::Parallel) ||
            "SIMD" >> pure(common::OmpIfClauseDirectiveNameModifier::Simd) ||
            "TARGET ENTER DATA" >>
                pure(common::OmpIfClauseDirectiveNameModifier::
                        TargetEnterData) ||
            "TARGET EXIT DATA" >>
                pure(
                    common::OmpIfClauseDirectiveNameModifier::TargetExitData) ||
            "TARGET DATA" >>
                pure(common::OmpIfClauseDirectiveNameModifier::TargetData) ||
            "TARGET UPDATE" >>
                pure(common::OmpIfClauseDirectiveNameModifier::TargetUpdate) ||
            "TARGET" >>
                pure(common::OmpIfClauseDirectiveNameModifier::Target) ||
            "TASK"_id >> pure(common::OmpIfClauseDirectiveNameModifier::Task) ||
            "TASKLOOP" >>
                pure(common::OmpIfClauseDirectiveNameModifier::Taskloop) ||
            "TEAMS" >> pure(common::OmpIfClauseDirectiveNameModifier::Teams)) /
        ":"),
    scalarLogicalExpr))

// 2.15.3.6 REDUCTION (reduction-identifier: variable-name-list)
TYPE_PARSER(construct<OmpReductionOperator>(Parser<DefinedOperator>{}) ||
    construct<OmpReductionOperator>(Parser<ProcedureDesignator>{}))

TYPE_PARSER(construct<OmpReductionClause>(
    Parser<OmpReductionOperator>{} / ":", Parser<OmpObjectList>{}))

// OMP 5.0 2.19.5.6 IN_REDUCTION (reduction-identifier: variable-name-list)
TYPE_PARSER(construct<OmpInReductionClause>(
    Parser<OmpReductionOperator>{} / ":", Parser<OmpObjectList>{}))

// OMP 5.0 2.11.4 allocate-clause -> ALLOCATE ([allocator:] variable-name-list)
// OMP 5.2 2.13.4 allocate-clause -> ALLOCATE ([allocate-modifier
//                                   [, allocate-modifier] :]
//                                   variable-name-list)
//                allocate-modifier -> allocator | align
TYPE_PARSER(construct<OmpAllocateClause>(
    maybe(
        first(
            construct<OmpAllocateClause::AllocateModifier>("ALLOCATOR" >>
                construct<OmpAllocateClause::AllocateModifier::ComplexModifier>(
                    parenthesized(construct<
                        OmpAllocateClause::AllocateModifier::Allocator>(
                        scalarIntExpr)) /
                        ",",
                    "ALIGN" >> parenthesized(construct<
                                   OmpAllocateClause::AllocateModifier::Align>(
                                   scalarIntExpr)))),
            construct<OmpAllocateClause::AllocateModifier>("ALLOCATOR" >>
                parenthesized(
                    construct<OmpAllocateClause::AllocateModifier::Allocator>(
                        scalarIntExpr))),
            construct<OmpAllocateClause::AllocateModifier>("ALIGN" >>
                parenthesized(
                    construct<OmpAllocateClause::AllocateModifier::Align>(
                        scalarIntExpr))),
            construct<OmpAllocateClause::AllocateModifier>(
                construct<OmpAllocateClause::AllocateModifier::Allocator>(
                    scalarIntExpr))) /
        ":"),
    Parser<OmpObjectList>{}))

// 2.13.9 DEPEND (SOURCE | SINK : vec | (IN | OUT | INOUT) : list
TYPE_PARSER(construct<OmpDependSinkVecLength>(
    Parser<DefinedOperator>{}, scalarIntConstantExpr))

TYPE_PARSER(
    construct<OmpDependSinkVec>(name, maybe(Parser<OmpDependSinkVecLength>{})))

TYPE_PARSER(construct<OmpDependenceType>(
    "IN"_id >> pure(common::OmpDependenceKind::In) ||
    "INOUT" >> pure(common::OmpDependenceKind::Inout) ||
    "OUT" >> pure(common::OmpDependenceKind::Out)))

TYPE_CONTEXT_PARSER("Omp Depend clause"_en_US,
    construct<OmpDependClause>(construct<OmpDependClause::Sink>(
        "SINK :" >> nonemptyList(Parser<OmpDependSinkVec>{}))) ||
        construct<OmpDependClause>(
            construct<OmpDependClause::Source>("SOURCE"_tok)) ||
        construct<OmpDependClause>(construct<OmpDependClause::InOut>(
            Parser<OmpDependenceType>{}, ":" >> nonemptyList(designator))))

// 2.15.3.7 LINEAR (linear-list: linear-step)
//          linear-list -> list | modifier(list)
//          linear-modifier -> REF | VAL | UVAL
TYPE_PARSER(construct<OmpLinearModifier>(
    "REF" >> pure(common::OmpLinearModifierKind::Ref) ||
    "VAL" >> pure(common::OmpLinearModifierKind::Val) ||
    "UVAL" >> pure(common::OmpLinearModifierKind::Uval)))

TYPE_CONTEXT_PARSER("Omp LINEAR clause"_en_US,
    construct<OmpLinearClause>(
        construct<OmpLinearClause>(construct<OmpLinearClause::WithModifier>(
            Parser<OmpLinearModifier>{}, parenthesized(nonemptyList(name)),
            maybe(":" >> scalarIntConstantExpr))) ||
        construct<OmpLinearClause>(construct<OmpLinearClause::WithoutModifier>(
            nonemptyList(name), maybe(":" >> scalarIntConstantExpr)))))

// 2.8.1 ALIGNED (list: alignment)
TYPE_PARSER(construct<OmpAlignedClause>(
    Parser<OmpObjectList>{}, maybe(":" >> scalarIntConstantExpr)))

// 2.9.5 ORDER ([order-modifier :]concurrent)
TYPE_PARSER(construct<OmpOrderModifier>("REPRODUCIBLE" >>
                pure(common::OmpOrderModifierKind::Reproducible)) ||
    construct<OmpOrderModifier>(
        "UNCONSTRAINED" >> pure(common::OmpOrderModifierKind::Unconstrained)))

TYPE_PARSER(construct<OmpOrderClause>(maybe(Parser<OmpOrderModifier>{} / ":"),
    "CONCURRENT" >> pure(common::OmpOrderClauseKind::Concurrent)))

TYPE_PARSER(
    construct<OmpObject>(designator) || construct<OmpObject>("/" >> name / "/"))

TYPE_PARSER(
    "ACQUIRE" >> construct<OmpClause>(construct<OmpClause::Acquire>()) ||
    "ACQ_REL" >> construct<OmpClause>(construct<OmpClause::AcqRel>()) ||
    "ALIGNED" >> construct<OmpClause>(construct<OmpClause::Aligned>(
                     parenthesized(Parser<OmpAlignedClause>{}))) ||
    "ALLOCATE" >> construct<OmpClause>(construct<OmpClause::Allocate>(
                      parenthesized(Parser<OmpAllocateClause>{}))) ||
    "ALLOCATOR" >> construct<OmpClause>(construct<OmpClause::Allocator>(
                       parenthesized(scalarIntExpr))) ||
    "ATOMIC_DEFAULT_MEM_ORDER" >>
        construct<OmpClause>(construct<OmpClause::AtomicDefaultMemOrder>(
            parenthesized(Parser<OmpAtomicDefaultMemOrderClause>{}))) ||
    "COLLAPSE" >> construct<OmpClause>(construct<OmpClause::Collapse>(
                      parenthesized(scalarIntConstantExpr))) ||
    "COPYIN" >> construct<OmpClause>(construct<OmpClause::Copyin>(
                    parenthesized(Parser<OmpObjectList>{}))) ||
    "COPYPRIVATE" >> construct<OmpClause>(construct<OmpClause::Copyprivate>(
                         (parenthesized(Parser<OmpObjectList>{})))) ||
    "DEFAULT"_id >> construct<OmpClause>(construct<OmpClause::Default>(
                        parenthesized(Parser<OmpDefaultClause>{}))) ||
    "DEFAULTMAP" >> construct<OmpClause>(construct<OmpClause::Defaultmap>(
                        parenthesized(Parser<OmpDefaultmapClause>{}))) ||
    "DEPEND" >> construct<OmpClause>(construct<OmpClause::Depend>(
                    parenthesized(Parser<OmpDependClause>{}))) ||
    "DEVICE" >> construct<OmpClause>(construct<OmpClause::Device>(
                    parenthesized(Parser<OmpDeviceClause>{}))) ||
    "DEVICE_TYPE" >> construct<OmpClause>(construct<OmpClause::DeviceType>(
                         parenthesized(Parser<OmpDeviceTypeClause>{}))) ||
    "DIST_SCHEDULE" >>
        construct<OmpClause>(construct<OmpClause::DistSchedule>(
            parenthesized("STATIC" >> maybe("," >> scalarIntExpr)))) ||
    "DYNAMIC_ALLOCATORS" >>
        construct<OmpClause>(construct<OmpClause::DynamicAllocators>()) ||
    "FINAL" >> construct<OmpClause>(construct<OmpClause::Final>(
                   parenthesized(scalarLogicalExpr))) ||
    "FULL" >> construct<OmpClause>(construct<OmpClause::Full>()) ||
    "FIRSTPRIVATE" >> construct<OmpClause>(construct<OmpClause::Firstprivate>(
                          parenthesized(Parser<OmpObjectList>{}))) ||
    "FROM" >> construct<OmpClause>(construct<OmpClause::From>(
                  parenthesized(Parser<OmpObjectList>{}))) ||
    "GRAINSIZE" >> construct<OmpClause>(construct<OmpClause::Grainsize>(
                       parenthesized(scalarIntExpr))) ||
    "HINT" >> construct<OmpClause>(
                  construct<OmpClause::Hint>(parenthesized(constantExpr))) ||
    "IF" >> construct<OmpClause>(construct<OmpClause::If>(
                parenthesized(Parser<OmpIfClause>{}))) ||
    "INBRANCH" >> construct<OmpClause>(construct<OmpClause::Inbranch>()) ||
    "IS_DEVICE_PTR" >> construct<OmpClause>(construct<OmpClause::IsDevicePtr>(
                           parenthesized(nonemptyList(name)))) ||
    "LASTPRIVATE" >> construct<OmpClause>(construct<OmpClause::Lastprivate>(
                         parenthesized(Parser<OmpObjectList>{}))) ||
    "LINEAR" >> construct<OmpClause>(construct<OmpClause::Linear>(
                    parenthesized(Parser<OmpLinearClause>{}))) ||
    "LINK" >> construct<OmpClause>(construct<OmpClause::Link>(
                  parenthesized(Parser<OmpObjectList>{}))) ||
    "MAP" >> construct<OmpClause>(construct<OmpClause::Map>(
                 parenthesized(Parser<OmpMapClause>{}))) ||
    "MERGEABLE" >> construct<OmpClause>(construct<OmpClause::Mergeable>()) ||
    "NOGROUP" >> construct<OmpClause>(construct<OmpClause::Nogroup>()) ||
    "NONTEMPORAL" >> construct<OmpClause>(construct<OmpClause::Nontemporal>(
                         parenthesized(nonemptyList(name)))) ||
    "NOTINBRANCH" >>
        construct<OmpClause>(construct<OmpClause::Notinbranch>()) ||
    "NOWAIT" >> construct<OmpClause>(construct<OmpClause::Nowait>()) ||
    "NUM_TASKS" >> construct<OmpClause>(construct<OmpClause::NumTasks>(
                       parenthesized(scalarIntExpr))) ||
    "NUM_TEAMS" >> construct<OmpClause>(construct<OmpClause::NumTeams>(
                       parenthesized(scalarIntExpr))) ||
    "NUM_THREADS" >> construct<OmpClause>(construct<OmpClause::NumThreads>(
                         parenthesized(scalarIntExpr))) ||
    "ORDER" >> construct<OmpClause>(construct<OmpClause::Order>(
                   parenthesized(Parser<OmpOrderClause>{}))) ||
    "ORDERED" >> construct<OmpClause>(construct<OmpClause::Ordered>(
                     maybe(parenthesized(scalarIntConstantExpr)))) ||
    "PARTIAL" >> construct<OmpClause>(construct<OmpClause::Partial>(
                     maybe(parenthesized(scalarIntConstantExpr)))) ||
    "PRIORITY" >> construct<OmpClause>(construct<OmpClause::Priority>(
                      parenthesized(scalarIntExpr))) ||
    "PRIVATE" >> construct<OmpClause>(construct<OmpClause::Private>(
                     parenthesized(Parser<OmpObjectList>{}))) ||
    "PROC_BIND" >> construct<OmpClause>(construct<OmpClause::ProcBind>(
                       parenthesized(Parser<OmpProcBindClause>{}))) ||
    "REDUCTION" >> construct<OmpClause>(construct<OmpClause::Reduction>(
                       parenthesized(Parser<OmpReductionClause>{}))) ||
    "IN_REDUCTION" >> construct<OmpClause>(construct<OmpClause::InReduction>(
                          parenthesized(Parser<OmpInReductionClause>{}))) ||
    "TASK_REDUCTION" >>
        construct<OmpClause>(construct<OmpClause::TaskReduction>(
            parenthesized(Parser<OmpReductionClause>{}))) ||
    "RELAXED" >> construct<OmpClause>(construct<OmpClause::Relaxed>()) ||
    "RELEASE" >> construct<OmpClause>(construct<OmpClause::Release>()) ||
    "REVERSE_OFFLOAD" >>
        construct<OmpClause>(construct<OmpClause::ReverseOffload>()) ||
    "SAFELEN" >> construct<OmpClause>(construct<OmpClause::Safelen>(
                     parenthesized(scalarIntConstantExpr))) ||
    "SCHEDULE" >> construct<OmpClause>(construct<OmpClause::Schedule>(
                      parenthesized(Parser<OmpScheduleClause>{}))) ||
    "SEQ_CST" >> construct<OmpClause>(construct<OmpClause::SeqCst>()) ||
    "SHARED" >> construct<OmpClause>(construct<OmpClause::Shared>(
                    parenthesized(Parser<OmpObjectList>{}))) ||
    "SIMD"_id >> construct<OmpClause>(construct<OmpClause::Simd>()) ||
    "SIMDLEN" >> construct<OmpClause>(construct<OmpClause::Simdlen>(
                     parenthesized(scalarIntConstantExpr))) ||
    "SIZES" >> construct<OmpClause>(construct<OmpClause::Sizes>(
                   parenthesized(nonemptyList(scalarIntExpr)))) ||
    "THREADS" >> construct<OmpClause>(construct<OmpClause::Threads>()) ||
    "THREAD_LIMIT" >> construct<OmpClause>(construct<OmpClause::ThreadLimit>(
                          parenthesized(scalarIntExpr))) ||
    "TO" >> construct<OmpClause>(construct<OmpClause::To>(
                parenthesized(Parser<OmpObjectList>{}))) ||
    "USE_DEVICE_PTR" >> construct<OmpClause>(construct<OmpClause::UseDevicePtr>(
                            parenthesized(Parser<OmpObjectList>{}))) ||
    "USE_DEVICE_ADDR" >>
        construct<OmpClause>(construct<OmpClause::UseDeviceAddr>(
            parenthesized(Parser<OmpObjectList>{}))) ||
    "UNIFIED_ADDRESS" >>
        construct<OmpClause>(construct<OmpClause::UnifiedAddress>()) ||
    "UNIFIED_SHARED_MEMORY" >>
        construct<OmpClause>(construct<OmpClause::UnifiedSharedMemory>()) ||
    "UNIFORM" >> construct<OmpClause>(construct<OmpClause::Uniform>(
                     parenthesized(nonemptyList(name)))) ||
    "UNTIED" >> construct<OmpClause>(construct<OmpClause::Untied>()))

// [Clause, [Clause], ...]
TYPE_PARSER(sourced(construct<OmpClauseList>(
    many(maybe(","_tok) >> sourced(Parser<OmpClause>{})))))

// 2.1 (variable | /common-block | array-sections)
TYPE_PARSER(construct<OmpObjectList>(nonemptyList(Parser<OmpObject>{})))

// Omp directives enclosing do loop
TYPE_PARSER(sourced(construct<OmpLoopDirective>(first(
    "DISTRIBUTE PARALLEL DO SIMD" >>
        pure(llvm::omp::Directive::OMPD_distribute_parallel_do_simd),
    "DISTRIBUTE PARALLEL DO" >>
        pure(llvm::omp::Directive::OMPD_distribute_parallel_do),
    "DISTRIBUTE SIMD" >> pure(llvm::omp::Directive::OMPD_distribute_simd),
    "DISTRIBUTE" >> pure(llvm::omp::Directive::OMPD_distribute),
    "DO SIMD" >> pure(llvm::omp::Directive::OMPD_do_simd),
    "DO" >> pure(llvm::omp::Directive::OMPD_do),
    "PARALLEL DO SIMD" >> pure(llvm::omp::Directive::OMPD_parallel_do_simd),
    "PARALLEL DO" >> pure(llvm::omp::Directive::OMPD_parallel_do),
    "SIMD" >> pure(llvm::omp::Directive::OMPD_simd),
    "TARGET PARALLEL DO SIMD" >>
        pure(llvm::omp::Directive::OMPD_target_parallel_do_simd),
    "TARGET PARALLEL DO" >> pure(llvm::omp::Directive::OMPD_target_parallel_do),
    "TARGET SIMD" >> pure(llvm::omp::Directive::OMPD_target_simd),
    "TARGET TEAMS DISTRIBUTE PARALLEL DO SIMD" >>
        pure(llvm::omp::Directive::
                OMPD_target_teams_distribute_parallel_do_simd),
    "TARGET TEAMS DISTRIBUTE PARALLEL DO" >>
        pure(llvm::omp::Directive::OMPD_target_teams_distribute_parallel_do),
    "TARGET TEAMS DISTRIBUTE SIMD" >>
        pure(llvm::omp::Directive::OMPD_target_teams_distribute_simd),
    "TARGET TEAMS DISTRIBUTE" >>
        pure(llvm::omp::Directive::OMPD_target_teams_distribute),
    "TASKLOOP SIMD" >> pure(llvm::omp::Directive::OMPD_taskloop_simd),
    "TASKLOOP" >> pure(llvm::omp::Directive::OMPD_taskloop),
    "TEAMS DISTRIBUTE PARALLEL DO SIMD" >>
        pure(llvm::omp::Directive::OMPD_teams_distribute_parallel_do_simd),
    "TEAMS DISTRIBUTE PARALLEL DO" >>
        pure(llvm::omp::Directive::OMPD_teams_distribute_parallel_do),
    "TEAMS DISTRIBUTE SIMD" >>
        pure(llvm::omp::Directive::OMPD_teams_distribute_simd),
    "TEAMS DISTRIBUTE" >> pure(llvm::omp::Directive::OMPD_teams_distribute),
    "TILE" >> pure(llvm::omp::Directive::OMPD_tile),
    "UNROLL" >> pure(llvm::omp::Directive::OMPD_unroll)))))

TYPE_PARSER(sourced(construct<OmpBeginLoopDirective>(
    sourced(Parser<OmpLoopDirective>{}), Parser<OmpClauseList>{})))

// 2.14.1 construct-type-clause -> PARALLEL | SECTIONS | DO | TASKGROUP
TYPE_PARSER(sourced(construct<OmpCancelType>(
    first("PARALLEL" >> pure(common::OmpCancelKind::Parallel),
        "SECTIONS" >> pure(common::OmpCancelKind::Sections),
        "DO" >> pure(common::OmpCancelKind::Do),
        "TASKGROUP" >> pure(common::OmpCancelKind::Taskgroup)))))

// 2.14.2 Cancellation Point construct
TYPE_PARSER(sourced(construct<OpenMPCancellationPointConstruct>(
    verbatim("CANCELLATION POINT"_tok), Parser<OmpCancelType>{})))

// 2.14.1 Cancel construct
TYPE_PARSER(sourced(construct<OpenMPCancelConstruct>(verbatim("CANCEL"_tok),
    Parser<OmpCancelType>{}, maybe("IF" >> parenthesized(scalarLogicalExpr)))))

// 2.17.7 Atomic construct/2.17.8 Flush construct [OpenMP 5.0]
//        memory-order-clause ->
//                               seq_cst
//                               acq_rel
//                               release
//                               acquire
//                               relaxed
TYPE_PARSER(sourced(construct<OmpMemoryOrderClause>(
    sourced("SEQ_CST" >> construct<OmpClause>(construct<OmpClause::SeqCst>()) ||
        "ACQ_REL" >> construct<OmpClause>(construct<OmpClause::AcqRel>()) ||
        "RELEASE" >> construct<OmpClause>(construct<OmpClause::Release>()) ||
        "ACQUIRE" >> construct<OmpClause>(construct<OmpClause::Acquire>()) ||
        "RELAXED" >> construct<OmpClause>(construct<OmpClause::Relaxed>())))))

// 2.4 Requires construct [OpenMP 5.0]
//        atomic-default-mem-order-clause ->
//                               seq_cst
//                               acq_rel
//                               relaxed
TYPE_PARSER(construct<OmpAtomicDefaultMemOrderClause>(
    "SEQ_CST" >> pure(common::OmpAtomicDefaultMemOrderClauseKind::SeqCst) ||
    "ACQ_REL" >> pure(common::OmpAtomicDefaultMemOrderClauseKind::AcqRel) ||
    "RELAXED" >> pure(common::OmpAtomicDefaultMemOrderClauseKind::Relaxed)))

// 2.17.7 Atomic construct
//        atomic-clause -> memory-order-clause | HINT(hint-expression)
TYPE_PARSER(sourced(construct<OmpAtomicClause>(
    construct<OmpAtomicClause>(Parser<OmpMemoryOrderClause>{}) ||
    construct<OmpAtomicClause>("HINT" >>
        sourced(construct<OmpClause>(
            construct<OmpClause::Hint>(parenthesized(constantExpr))))))))

// atomic-clause-list -> [atomic-clause, [atomic-clause], ...]
TYPE_PARSER(sourced(construct<OmpAtomicClauseList>(
    many(maybe(","_tok) >> sourced(Parser<OmpAtomicClause>{})))))

TYPE_PARSER(sourced(construct<OpenMPFlushConstruct>(verbatim("FLUSH"_tok),
    many(maybe(","_tok) >> sourced(Parser<OmpMemoryOrderClause>{})),
    maybe(parenthesized(Parser<OmpObjectList>{})))))

// Simple Standalone Directives
TYPE_PARSER(sourced(construct<OmpSimpleStandaloneDirective>(first(
    "BARRIER" >> pure(llvm::omp::Directive::OMPD_barrier),
    "ORDERED" >> pure(llvm::omp::Directive::OMPD_ordered),
    "TARGET ENTER DATA" >> pure(llvm::omp::Directive::OMPD_target_enter_data),
    "TARGET EXIT DATA" >> pure(llvm::omp::Directive::OMPD_target_exit_data),
    "TARGET UPDATE" >> pure(llvm::omp::Directive::OMPD_target_update),
    "TASKWAIT" >> pure(llvm::omp::Directive::OMPD_taskwait),
    "TASKYIELD" >> pure(llvm::omp::Directive::OMPD_taskyield)))))

TYPE_PARSER(sourced(construct<OpenMPSimpleStandaloneConstruct>(
    Parser<OmpSimpleStandaloneDirective>{}, Parser<OmpClauseList>{})))

// Standalone Constructs
TYPE_PARSER(
    sourced(construct<OpenMPStandaloneConstruct>(
                Parser<OpenMPSimpleStandaloneConstruct>{}) ||
        construct<OpenMPStandaloneConstruct>(Parser<OpenMPFlushConstruct>{}) ||
        construct<OpenMPStandaloneConstruct>(Parser<OpenMPCancelConstruct>{}) ||
        construct<OpenMPStandaloneConstruct>(
            Parser<OpenMPCancellationPointConstruct>{})) /
    endOfLine)

// Directives enclosing structured-block
TYPE_PARSER(construct<OmpBlockDirective>(first(
    "MASTER" >> pure(llvm::omp::Directive::OMPD_master),
    "ORDERED" >> pure(llvm::omp::Directive::OMPD_ordered),
    "PARALLEL WORKSHARE" >> pure(llvm::omp::Directive::OMPD_parallel_workshare),
    "PARALLEL" >> pure(llvm::omp::Directive::OMPD_parallel),
    "SINGLE" >> pure(llvm::omp::Directive::OMPD_single),
    "TARGET DATA" >> pure(llvm::omp::Directive::OMPD_target_data),
    "TARGET PARALLEL" >> pure(llvm::omp::Directive::OMPD_target_parallel),
    "TARGET TEAMS" >> pure(llvm::omp::Directive::OMPD_target_teams),
    "TARGET" >> pure(llvm::omp::Directive::OMPD_target),
    "TASK"_id >> pure(llvm::omp::Directive::OMPD_task),
    "TASKGROUP" >> pure(llvm::omp::Directive::OMPD_taskgroup),
    "TEAMS" >> pure(llvm::omp::Directive::OMPD_teams),
    "WORKSHARE" >> pure(llvm::omp::Directive::OMPD_workshare))))

TYPE_PARSER(sourced(construct<OmpBeginBlockDirective>(
    sourced(Parser<OmpBlockDirective>{}), Parser<OmpClauseList>{})))

TYPE_PARSER(construct<OmpReductionInitializerClause>(
    "INITIALIZER" >> parenthesized("OMP_PRIV =" >> expr)))

// 2.16 Declare Reduction Construct
TYPE_PARSER(sourced(construct<OpenMPDeclareReductionConstruct>(
    verbatim("DECLARE REDUCTION"_tok),
    "(" >> Parser<OmpReductionOperator>{} / ":",
    nonemptyList(Parser<DeclarationTypeSpec>{}) / ":",
    Parser<OmpReductionCombiner>{} / ")",
    maybe(Parser<OmpReductionInitializerClause>{}))))

// declare-target with list
TYPE_PARSER(sourced(construct<OmpDeclareTargetWithList>(
    parenthesized(Parser<OmpObjectList>{}))))

// declare-target with clause
TYPE_PARSER(
    sourced(construct<OmpDeclareTargetWithClause>(Parser<OmpClauseList>{})))

// declare-target-specifier
TYPE_PARSER(
    construct<OmpDeclareTargetSpecifier>(Parser<OmpDeclareTargetWithList>{}) ||
    construct<OmpDeclareTargetSpecifier>(Parser<OmpDeclareTargetWithClause>{}))

// 2.10.6 Declare Target Construct
TYPE_PARSER(sourced(construct<OpenMPDeclareTargetConstruct>(
    verbatim("DECLARE TARGET"_tok), Parser<OmpDeclareTargetSpecifier>{})))

TYPE_PARSER(construct<OmpReductionCombiner>(Parser<AssignmentStmt>{}) ||
    construct<OmpReductionCombiner>(
        construct<OmpReductionCombiner::FunctionCombiner>(
            construct<Call>(Parser<ProcedureDesignator>{},
                parenthesized(optionalList(actualArgSpec))))))

// 2.17.7 atomic -> ATOMIC [clause [,]] atomic-clause [[,] clause] |
//                  ATOMIC [clause]
//       clause -> memory-order-clause | HINT(hint-expression)
//       memory-order-clause -> SEQ_CST | ACQ_REL | RELEASE | ACQUIRE | RELAXED
//       atomic-clause -> READ | WRITE | UPDATE | CAPTURE

// OMP END ATOMIC
TYPE_PARSER(construct<OmpEndAtomic>(startOmpLine >> "END ATOMIC"_tok))

// OMP ATOMIC [MEMORY-ORDER-CLAUSE-LIST] READ [MEMORY-ORDER-CLAUSE-LIST]
TYPE_PARSER("ATOMIC" >>
    construct<OmpAtomicRead>(Parser<OmpAtomicClauseList>{} / maybe(","_tok),
        verbatim("READ"_tok), Parser<OmpAtomicClauseList>{} / endOmpLine,
        statement(assignmentStmt), maybe(Parser<OmpEndAtomic>{} / endOmpLine)))

// OMP ATOMIC [MEMORY-ORDER-CLAUSE-LIST] CAPTURE [MEMORY-ORDER-CLAUSE-LIST]
TYPE_PARSER("ATOMIC" >>
    construct<OmpAtomicCapture>(Parser<OmpAtomicClauseList>{} / maybe(","_tok),
        verbatim("CAPTURE"_tok), Parser<OmpAtomicClauseList>{} / endOmpLine,
        statement(assignmentStmt), statement(assignmentStmt),
        Parser<OmpEndAtomic>{} / endOmpLine))

// OMP ATOMIC [MEMORY-ORDER-CLAUSE-LIST] UPDATE [MEMORY-ORDER-CLAUSE-LIST]
TYPE_PARSER("ATOMIC" >>
    construct<OmpAtomicUpdate>(Parser<OmpAtomicClauseList>{} / maybe(","_tok),
        verbatim("UPDATE"_tok), Parser<OmpAtomicClauseList>{} / endOmpLine,
        statement(assignmentStmt), maybe(Parser<OmpEndAtomic>{} / endOmpLine)))

// OMP ATOMIC [atomic-clause-list]
TYPE_PARSER(construct<OmpAtomic>(verbatim("ATOMIC"_tok),
    Parser<OmpAtomicClauseList>{} / endOmpLine, statement(assignmentStmt),
    maybe(Parser<OmpEndAtomic>{} / endOmpLine)))

// OMP ATOMIC [MEMORY-ORDER-CLAUSE-LIST] WRITE [MEMORY-ORDER-CLAUSE-LIST]
TYPE_PARSER("ATOMIC" >>
    construct<OmpAtomicWrite>(Parser<OmpAtomicClauseList>{} / maybe(","_tok),
        verbatim("WRITE"_tok), Parser<OmpAtomicClauseList>{} / endOmpLine,
        statement(assignmentStmt), maybe(Parser<OmpEndAtomic>{} / endOmpLine)))

// Atomic Construct
TYPE_PARSER(construct<OpenMPAtomicConstruct>(Parser<OmpAtomicRead>{}) ||
    construct<OpenMPAtomicConstruct>(Parser<OmpAtomicCapture>{}) ||
    construct<OpenMPAtomicConstruct>(Parser<OmpAtomicWrite>{}) ||
    construct<OpenMPAtomicConstruct>(Parser<OmpAtomicUpdate>{}) ||
    construct<OpenMPAtomicConstruct>(Parser<OmpAtomic>{}))

// 2.13.2 OMP CRITICAL
TYPE_PARSER(startOmpLine >>
    sourced(construct<OmpEndCriticalDirective>(
        verbatim("END CRITICAL"_tok), maybe(parenthesized(name)))) /
        endOmpLine)
TYPE_PARSER(sourced(construct<OmpCriticalDirective>(verbatim("CRITICAL"_tok),
                maybe(parenthesized(name)), Parser<OmpClauseList>{})) /
    endOmpLine)

TYPE_PARSER(construct<OpenMPCriticalConstruct>(
    Parser<OmpCriticalDirective>{}, block, Parser<OmpEndCriticalDirective>{}))

// 2.11.3 Executable Allocate directive
TYPE_PARSER(
    sourced(construct<OpenMPExecutableAllocate>(verbatim("ALLOCATE"_tok),
        maybe(parenthesized(Parser<OmpObjectList>{})), Parser<OmpClauseList>{},
        maybe(nonemptyList(Parser<OpenMPDeclarativeAllocate>{})) / endOmpLine,
        statement(allocateStmt))))

// 6.7 Allocators construct [OpenMP 5.2]
//     allocators-construct -> ALLOCATORS [allocate-clause [,]]
//                                allocate-stmt
//                             [omp-end-allocators-construct]
TYPE_PARSER(sourced(construct<OpenMPAllocatorsConstruct>(
    verbatim("ALLOCATORS"_tok), Parser<OmpClauseList>{} / endOmpLine,
    statement(allocateStmt), maybe(Parser<OmpEndAllocators>{} / endOmpLine))))

TYPE_PARSER(construct<OmpEndAllocators>(startOmpLine >> "END ALLOCATORS"_tok))

// 2.8.2 Declare Simd construct
TYPE_PARSER(
    sourced(construct<OpenMPDeclareSimdConstruct>(verbatim("DECLARE SIMD"_tok),
        maybe(parenthesized(name)), Parser<OmpClauseList>{})))

// 2.4 Requires construct
TYPE_PARSER(sourced(construct<OpenMPRequiresConstruct>(
    verbatim("REQUIRES"_tok), Parser<OmpClauseList>{})))

// 2.15.2 Threadprivate directive
TYPE_PARSER(sourced(construct<OpenMPThreadprivate>(
    verbatim("THREADPRIVATE"_tok), parenthesized(Parser<OmpObjectList>{}))))

// 2.11.3 Declarative Allocate directive
TYPE_PARSER(
    sourced(construct<OpenMPDeclarativeAllocate>(verbatim("ALLOCATE"_tok),
        parenthesized(Parser<OmpObjectList>{}), Parser<OmpClauseList>{})) /
    lookAhead(endOmpLine / !statement(allocateStmt)))

// Declarative constructs
TYPE_PARSER(startOmpLine >>
    sourced(construct<OpenMPDeclarativeConstruct>(
                Parser<OpenMPDeclareReductionConstruct>{}) ||
        construct<OpenMPDeclarativeConstruct>(
            Parser<OpenMPDeclareSimdConstruct>{}) ||
        construct<OpenMPDeclarativeConstruct>(
            Parser<OpenMPDeclareTargetConstruct>{}) ||
        construct<OpenMPDeclarativeConstruct>(
            Parser<OpenMPDeclarativeAllocate>{}) ||
        construct<OpenMPDeclarativeConstruct>(
            Parser<OpenMPRequiresConstruct>{}) ||
        construct<OpenMPDeclarativeConstruct>(Parser<OpenMPThreadprivate>{})) /
        endOmpLine)

// Block Construct
TYPE_PARSER(construct<OpenMPBlockConstruct>(
    Parser<OmpBeginBlockDirective>{} / endOmpLine, block,
    Parser<OmpEndBlockDirective>{} / endOmpLine))

// OMP SECTIONS Directive
TYPE_PARSER(construct<OmpSectionsDirective>(first(
    "SECTIONS" >> pure(llvm::omp::Directive::OMPD_sections),
    "PARALLEL SECTIONS" >> pure(llvm::omp::Directive::OMPD_parallel_sections))))

// OMP BEGIN and END SECTIONS Directive
TYPE_PARSER(sourced(construct<OmpBeginSectionsDirective>(
    sourced(Parser<OmpSectionsDirective>{}), Parser<OmpClauseList>{})))
TYPE_PARSER(
    startOmpLine >> sourced(construct<OmpEndSectionsDirective>(
                        sourced("END"_tok >> Parser<OmpSectionsDirective>{}),
                        Parser<OmpClauseList>{})))

// OMP SECTION-BLOCK

TYPE_PARSER(construct<OpenMPSectionConstruct>(block))

TYPE_PARSER(maybe(startOmpLine >> "SECTION"_tok / endOmpLine) >>
    construct<OmpSectionBlocks>(nonemptySeparated(
        construct<OpenMPConstruct>(sourced(Parser<OpenMPSectionConstruct>{})),
        startOmpLine >> "SECTION"_tok / endOmpLine)))

// OMP SECTIONS (OpenMP 5.0 - 2.8.1), PARALLEL SECTIONS (OpenMP 5.0 - 2.13.3)
TYPE_PARSER(construct<OpenMPSectionsConstruct>(
    Parser<OmpBeginSectionsDirective>{} / endOmpLine,
    Parser<OmpSectionBlocks>{}, Parser<OmpEndSectionsDirective>{} / endOmpLine))

TYPE_CONTEXT_PARSER("OpenMP construct"_en_US,
    startOmpLine >>
        first(construct<OpenMPConstruct>(Parser<OpenMPSectionsConstruct>{}),
            construct<OpenMPConstruct>(Parser<OpenMPLoopConstruct>{}),
            construct<OpenMPConstruct>(Parser<OpenMPBlockConstruct>{}),
            // OpenMPBlockConstruct is attempted before
            // OpenMPStandaloneConstruct to resolve !$OMP ORDERED
            construct<OpenMPConstruct>(Parser<OpenMPStandaloneConstruct>{}),
            construct<OpenMPConstruct>(Parser<OpenMPAtomicConstruct>{}),
            construct<OpenMPConstruct>(Parser<OpenMPExecutableAllocate>{}),
            construct<OpenMPConstruct>(Parser<OpenMPAllocatorsConstruct>{}),
            construct<OpenMPConstruct>(Parser<OpenMPDeclarativeAllocate>{}),
            construct<OpenMPConstruct>(Parser<OpenMPCriticalConstruct>{})))

// END OMP Block directives
TYPE_PARSER(
    startOmpLine >> sourced(construct<OmpEndBlockDirective>(
                        sourced("END"_tok >> Parser<OmpBlockDirective>{}),
                        Parser<OmpClauseList>{})))

// END OMP Loop directives
TYPE_PARSER(
    startOmpLine >> sourced(construct<OmpEndLoopDirective>(
                        sourced("END"_tok >> Parser<OmpLoopDirective>{}),
                        Parser<OmpClauseList>{})))

TYPE_PARSER(construct<OpenMPLoopConstruct>(
    Parser<OmpBeginLoopDirective>{} / endOmpLine))
} // namespace Fortran::parser
