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
    "PRIVATE" >> pure(OmpDefaultClause::Type::Private) ||
    "FIRSTPRIVATE" >> pure(OmpDefaultClause::Type::Firstprivate) ||
    "SHARED" >> pure(OmpDefaultClause::Type::Shared) ||
    "NONE" >> pure(OmpDefaultClause::Type::None)))

// 2.5 PROC_BIND (MASTER | CLOSE | SPREAD)
TYPE_PARSER(construct<OmpProcBindClause>(
    "CLOSE" >> pure(OmpProcBindClause::Type::Close) ||
    "MASTER" >> pure(OmpProcBindClause::Type::Master) ||
    "SPREAD" >> pure(OmpProcBindClause::Type::Spread)))

// 2.15.5.1 MAP ([ [ALWAYS[,]] map-type : ] variable-name-list)
//          map-type -> TO | FROM | TOFROM | ALLOC | RELEASE | DELETE
TYPE_PARSER(construct<OmpMapType>(
    maybe("ALWAYS" >> construct<OmpMapType::Always>() / maybe(","_tok)),
    ("TO"_id >> pure(OmpMapType::Type::To) ||
        "FROM" >> pure(OmpMapType::Type::From) ||
        "TOFROM" >> pure(OmpMapType::Type::Tofrom) ||
        "ALLOC" >> pure(OmpMapType::Type::Alloc) ||
        "RELEASE" >> pure(OmpMapType::Type::Release) ||
        "DELETE" >> pure(OmpMapType::Type::Delete)) /
        ":"))

TYPE_PARSER(construct<OmpMapClause>(
    maybe(Parser<OmpMapType>{}), Parser<OmpObjectList>{}))

// 2.15.5.2 defaultmap -> DEFAULTMAP (TOFROM:SCALAR)
TYPE_PARSER(construct<OmpDefaultmapClause>(
    construct<OmpDefaultmapClause::ImplicitBehavior>(
        "TOFROM" >> pure(OmpDefaultmapClause::ImplicitBehavior::Tofrom)),
    maybe(":" >> construct<OmpDefaultmapClause::VariableCategory>("SCALAR" >>
                     pure(OmpDefaultmapClause::VariableCategory::Scalar)))))

// 2.7.1 SCHEDULE ([modifier1 [, modifier2]:]kind[, chunk_size])
//       Modifier ->  MONITONIC | NONMONOTONIC | SIMD
//       kind -> STATIC | DYNAMIC | GUIDED | AUTO | RUNTIME
//       chunk_size -> ScalarIntExpr
TYPE_PARSER(construct<OmpScheduleModifierType>(
    "MONOTONIC" >> pure(OmpScheduleModifierType::ModType::Monotonic) ||
    "NONMONOTONIC" >> pure(OmpScheduleModifierType::ModType::Nonmonotonic) ||
    "SIMD" >> pure(OmpScheduleModifierType::ModType::Simd)))

TYPE_PARSER(construct<OmpScheduleModifier>(Parser<OmpScheduleModifierType>{},
    maybe("," >> Parser<OmpScheduleModifierType>{}) / ":"))

TYPE_PARSER(construct<OmpScheduleClause>(maybe(Parser<OmpScheduleModifier>{}),
    "STATIC" >> pure(OmpScheduleClause::ScheduleType::Static) ||
        "DYNAMIC" >> pure(OmpScheduleClause::ScheduleType::Dynamic) ||
        "GUIDED" >> pure(OmpScheduleClause::ScheduleType::Guided) ||
        "AUTO" >> pure(OmpScheduleClause::ScheduleType::Auto) ||
        "RUNTIME" >> pure(OmpScheduleClause::ScheduleType::Runtime),
    maybe("," >> scalarIntExpr)))

// 2.12 IF (directive-name-modifier: scalar-logical-expr)
TYPE_PARSER(construct<OmpIfClause>(
    maybe(
        ("PARALLEL" >> pure(OmpIfClause::DirectiveNameModifier::Parallel) ||
            "TARGET ENTER DATA" >>
                pure(OmpIfClause::DirectiveNameModifier::TargetEnterData) ||
            "TARGET EXIT DATA" >>
                pure(OmpIfClause::DirectiveNameModifier::TargetExitData) ||
            "TARGET DATA" >>
                pure(OmpIfClause::DirectiveNameModifier::TargetData) ||
            "TARGET UPDATE" >>
                pure(OmpIfClause::DirectiveNameModifier::TargetUpdate) ||
            "TARGET" >> pure(OmpIfClause::DirectiveNameModifier::Target) ||
            "TASK"_id >> pure(OmpIfClause::DirectiveNameModifier::Task) ||
            "TASKLOOP" >> pure(OmpIfClause::DirectiveNameModifier::Taskloop)) /
        ":"),
    scalarLogicalExpr))

// 2.15.3.6 REDUCTION (reduction-identifier: variable-name-list)
TYPE_PARSER(construct<OmpReductionOperator>(Parser<DefinedOperator>{}) ||
    construct<OmpReductionOperator>(Parser<ProcedureDesignator>{}))

TYPE_PARSER(construct<OmpReductionClause>(
    Parser<OmpReductionOperator>{} / ":", nonemptyList(designator)))

// OMP 5.0 2.11.4  ALLOCATE ([allocator:] variable-name-list)
TYPE_PARSER(construct<OmpAllocateClause>(
    maybe(construct<OmpAllocateClause::Allocator>(scalarIntExpr) / ":"),
    Parser<OmpObjectList>{}))

// 2.13.9 DEPEND (SOURCE | SINK : vec | (IN | OUT | INOUT) : list
TYPE_PARSER(construct<OmpDependSinkVecLength>(
    Parser<DefinedOperator>{}, scalarIntConstantExpr))

TYPE_PARSER(
    construct<OmpDependSinkVec>(name, maybe(Parser<OmpDependSinkVecLength>{})))

TYPE_PARSER(
    construct<OmpDependenceType>("IN"_id >> pure(OmpDependenceType::Type::In) ||
        "INOUT" >> pure(OmpDependenceType::Type::Inout) ||
        "OUT" >> pure(OmpDependenceType::Type::Out)))

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
TYPE_PARSER(
    construct<OmpLinearModifier>("REF" >> pure(OmpLinearModifier::Type::Ref) ||
        "VAL" >> pure(OmpLinearModifier::Type::Val) ||
        "UVAL" >> pure(OmpLinearModifier::Type::Uval)))

TYPE_CONTEXT_PARSER("Omp LINEAR clause"_en_US,
    construct<OmpLinearClause>(
        construct<OmpLinearClause>(construct<OmpLinearClause::WithModifier>(
            Parser<OmpLinearModifier>{}, parenthesized(nonemptyList(name)),
            maybe(":" >> scalarIntConstantExpr))) ||
        construct<OmpLinearClause>(construct<OmpLinearClause::WithoutModifier>(
            nonemptyList(name), maybe(":" >> scalarIntConstantExpr)))))

// 2.8.1 ALIGNED (list: alignment)
TYPE_PARSER(construct<OmpAlignedClause>(
    nonemptyList(name), maybe(":" >> scalarIntConstantExpr)))

TYPE_PARSER(
    construct<OmpObject>(designator) || construct<OmpObject>("/" >> name / "/"))

TYPE_PARSER("ALIGNED" >>
        construct<OmpClause>(parenthesized(Parser<OmpAlignedClause>{})) ||
    "COLLAPSE" >> construct<OmpClause>(construct<OmpClause::Collapse>(
                      parenthesized(scalarIntConstantExpr))) ||
    "COPYIN" >> construct<OmpClause>(construct<OmpClause::Copyin>(
                    parenthesized(Parser<OmpObjectList>{}))) ||
    "COPYPRIVATE" >> construct<OmpClause>(construct<OmpClause::Copyprivate>(
                         (parenthesized(Parser<OmpObjectList>{})))) ||
    "DEFAULT"_id >>
        construct<OmpClause>(parenthesized(Parser<OmpDefaultClause>{})) ||
    "DEFAULTMAP" >>
        construct<OmpClause>(parenthesized(Parser<OmpDefaultmapClause>{})) ||
    "DEPEND" >>
        construct<OmpClause>(parenthesized(Parser<OmpDependClause>{})) ||
    "DEVICE" >> construct<OmpClause>(construct<OmpClause::Device>(
                    parenthesized(scalarIntExpr))) ||
    "DIST_SCHEDULE" >>
        construct<OmpClause>(construct<OmpDistScheduleClause>(
            parenthesized("STATIC" >> maybe("," >> scalarIntExpr)))) ||
    "FINAL" >> construct<OmpClause>(construct<OmpClause::Final>(
                   parenthesized(scalarLogicalExpr))) ||
    "FIRSTPRIVATE" >> construct<OmpClause>(construct<OmpClause::Firstprivate>(
                          parenthesized(Parser<OmpObjectList>{}))) ||
    "FROM" >> construct<OmpClause>(construct<OmpClause::From>(
                  parenthesized(Parser<OmpObjectList>{}))) ||
    "GRAINSIZE" >> construct<OmpClause>(construct<OmpClause::Grainsize>(
                       parenthesized(scalarIntExpr))) ||
    "IF" >> construct<OmpClause>(parenthesized(Parser<OmpIfClause>{})) ||
    "INBRANCH" >> construct<OmpClause>(construct<OmpClause::Inbranch>()) ||
    "IS_DEVICE_PTR" >> construct<OmpClause>(construct<OmpClause::IsDevicePtr>(
                           parenthesized(nonemptyList(name)))) ||
    "LASTPRIVATE" >> construct<OmpClause>(construct<OmpClause::Lastprivate>(
                         parenthesized(Parser<OmpObjectList>{}))) ||
    "LINEAR" >>
        construct<OmpClause>(parenthesized(Parser<OmpLinearClause>{})) ||
    "LINK" >> construct<OmpClause>(construct<OmpClause::Link>(
                  parenthesized(Parser<OmpObjectList>{}))) ||
    "MAP" >> construct<OmpClause>(parenthesized(Parser<OmpMapClause>{})) ||
    "MERGEABLE" >> construct<OmpClause>(construct<OmpClause::Mergeable>()) ||
    "NOGROUP" >> construct<OmpClause>(construct<OmpClause::Nogroup>()) ||
    "NOTINBRANCH" >>
        construct<OmpClause>(construct<OmpClause::Notinbranch>()) ||
    "NOWAIT" >> construct<OmpClause>(construct<OmpNowait>()) ||
    "NUM_TASKS" >> construct<OmpClause>(construct<OmpClause::NumTasks>(
                       parenthesized(scalarIntExpr))) ||
    "NUM_TEAMS" >> construct<OmpClause>(construct<OmpClause::NumTeams>(
                       parenthesized(scalarIntExpr))) ||
    "NUM_THREADS" >> construct<OmpClause>(construct<OmpClause::NumThreads>(
                         parenthesized(scalarIntExpr))) ||
    "ORDERED" >> construct<OmpClause>(construct<OmpClause::Ordered>(
                     maybe(parenthesized(scalarIntConstantExpr)))) ||
    "PRIORITY" >> construct<OmpClause>(construct<OmpClause::Priority>(
                      parenthesized(scalarIntExpr))) ||
    "PRIVATE" >> construct<OmpClause>(construct<OmpClause::Private>(
                     parenthesized(Parser<OmpObjectList>{}))) ||
    "PROC_BIND" >>
        construct<OmpClause>(parenthesized(Parser<OmpProcBindClause>{})) ||
    "REDUCTION" >>
        construct<OmpClause>(parenthesized(Parser<OmpReductionClause>{})) ||
    "ALLOCATE" >>
        construct<OmpClause>(parenthesized(Parser<OmpAllocateClause>{})) ||
    "SAFELEN" >> construct<OmpClause>(construct<OmpClause::Safelen>(
                     parenthesized(scalarIntConstantExpr))) ||
    "SCHEDULE" >>
        construct<OmpClause>(parenthesized(Parser<OmpScheduleClause>{})) ||
    "SHARED" >> construct<OmpClause>(construct<OmpClause::Shared>(
                    parenthesized(Parser<OmpObjectList>{}))) ||
    "SIMD"_id >> construct<OmpClause>(construct<OmpClause::Simd>()) ||
    "SIMDLEN" >> construct<OmpClause>(construct<OmpClause::Simdlen>(
                     parenthesized(scalarIntConstantExpr))) ||
    "THREADS" >> construct<OmpClause>(construct<OmpClause::Threads>()) ||
    "THREAD_LIMIT" >> construct<OmpClause>(construct<OmpClause::ThreadLimit>(
                          parenthesized(scalarIntExpr))) ||
    "TO" >> construct<OmpClause>(construct<OmpClause::To>(
                parenthesized(Parser<OmpObjectList>{}))) ||
    "USE_DEVICE_PTR" >> construct<OmpClause>(construct<OmpClause::UseDevicePtr>(
                            parenthesized(nonemptyList(name)))) ||
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
    "TEAMS DISTRIBUTE" >> pure(llvm::omp::Directive::OMPD_teams_distribute)))))

TYPE_PARSER(sourced(construct<OmpBeginLoopDirective>(
    sourced(Parser<OmpLoopDirective>{}), Parser<OmpClauseList>{})))

// 2.14.1 construct-type-clause -> PARALLEL | SECTIONS | DO | TASKGROUP
TYPE_PARSER(sourced(construct<OmpCancelType>(
    first("PARALLEL" >> pure(OmpCancelType::Type::Parallel),
        "SECTIONS" >> pure(OmpCancelType::Type::Sections),
        "DO" >> pure(OmpCancelType::Type::Do),
        "TASKGROUP" >> pure(OmpCancelType::Type::Taskgroup)))))

// 2.14.2 Cancellation Point construct
TYPE_PARSER(sourced(construct<OpenMPCancellationPointConstruct>(
    verbatim("CANCELLATION POINT"_tok), Parser<OmpCancelType>{})))

// 2.14.1 Cancel construct
TYPE_PARSER(sourced(construct<OpenMPCancelConstruct>(verbatim("CANCEL"_tok),
    Parser<OmpCancelType>{}, maybe("IF" >> parenthesized(scalarLogicalExpr)))))

// 2.17.8 Flush construct [OpenMP 5.0]
//        flush -> FLUSH [memory-order-clause] [(variable-name-list)]
//        memory-order-clause -> acq_rel
//                               release
//                               acquire
TYPE_PARSER(sourced(construct<OmpFlushMemoryClause>(
    "ACQ_REL" >> pure(OmpFlushMemoryClause::FlushMemoryOrder::AcqRel) ||
    "RELEASE" >> pure(OmpFlushMemoryClause::FlushMemoryOrder::Release) ||
    "ACQUIRE" >> pure(OmpFlushMemoryClause::FlushMemoryOrder::Acquire))))

TYPE_PARSER(sourced(construct<OpenMPFlushConstruct>(verbatim("FLUSH"_tok),
    maybe(Parser<OmpFlushMemoryClause>{}),
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

// 2.13.6 ATOMIC [seq_cst[,]] atomic-clause [[,]seq_cst] | ATOMIC [seq_cst]
//        atomic-clause -> READ | WRITE | UPDATE | CAPTURE

// OMP END ATOMIC
TYPE_PARSER(construct<OmpEndAtomic>(startOmpLine >> "END ATOMIC"_tok))

// ATOMIC Memory related clause
TYPE_PARSER(sourced(construct<OmpMemoryClause>(
    "SEQ_CST" >> pure(OmpMemoryClause::MemoryOrder::SeqCst))))

// ATOMIC Memory Clause List
TYPE_PARSER(construct<OmpMemoryClauseList>(
    many(maybe(","_tok) >> Parser<OmpMemoryClause>{})))

TYPE_PARSER(construct<OmpMemoryClausePostList>(
    many(maybe(","_tok) >> Parser<OmpMemoryClause>{})))

// OMP [SEQ_CST] ATOMIC READ [SEQ_CST]
TYPE_PARSER("ATOMIC" >>
    construct<OmpAtomicRead>(Parser<OmpMemoryClauseList>{} / maybe(","_tok),
        verbatim("READ"_tok), Parser<OmpMemoryClausePostList>{} / endOmpLine,
        statement(assignmentStmt), maybe(Parser<OmpEndAtomic>{} / endOmpLine)))

// OMP ATOMIC [SEQ_CST] CAPTURE [SEQ_CST]
TYPE_PARSER("ATOMIC" >>
    construct<OmpAtomicCapture>(Parser<OmpMemoryClauseList>{} / maybe(","_tok),
        verbatim("CAPTURE"_tok), Parser<OmpMemoryClausePostList>{} / endOmpLine,
        statement(assignmentStmt), statement(assignmentStmt),
        Parser<OmpEndAtomic>{} / endOmpLine))

// OMP ATOMIC [SEQ_CST] UPDATE [SEQ_CST]
TYPE_PARSER("ATOMIC" >>
    construct<OmpAtomicUpdate>(Parser<OmpMemoryClauseList>{} / maybe(","_tok),
        verbatim("UPDATE"_tok), Parser<OmpMemoryClausePostList>{} / endOmpLine,
        statement(assignmentStmt), maybe(Parser<OmpEndAtomic>{} / endOmpLine)))

// OMP ATOMIC [SEQ_CST]
TYPE_PARSER(construct<OmpAtomic>(verbatim("ATOMIC"_tok),
    Parser<OmpMemoryClauseList>{} / endOmpLine, statement(assignmentStmt),
    maybe(Parser<OmpEndAtomic>{} / endOmpLine)))

// ATOMIC [SEQ_CST] WRITE [SEQ_CST]
TYPE_PARSER("ATOMIC" >>
    construct<OmpAtomicWrite>(Parser<OmpMemoryClauseList>{} / maybe(","_tok),
        verbatim("WRITE"_tok), Parser<OmpMemoryClausePostList>{} / endOmpLine,
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
                maybe(parenthesized(name)),
                maybe("HINT" >> construct<OmpCriticalDirective::Hint>(
                                    parenthesized(constantExpr))))) /
    endOmpLine)

TYPE_PARSER(construct<OpenMPCriticalConstruct>(
    Parser<OmpCriticalDirective>{}, block, Parser<OmpEndCriticalDirective>{}))

// 2.8.2 Declare Simd construct
TYPE_PARSER(
    sourced(construct<OpenMPDeclareSimdConstruct>(verbatim("DECLARE SIMD"_tok),
        maybe(parenthesized(name)), Parser<OmpClauseList>{})))

// 2.15.2 Threadprivate directive
TYPE_PARSER(sourced(construct<OpenMPThreadprivate>(
    verbatim("THREADPRIVATE"_tok), parenthesized(Parser<OmpObjectList>{}))))

// Declarative constructs
TYPE_PARSER(startOmpLine >>
    sourced(construct<OpenMPDeclarativeConstruct>(
                Parser<OpenMPDeclareReductionConstruct>{}) ||
        construct<OpenMPDeclarativeConstruct>(
            Parser<OpenMPDeclareSimdConstruct>{}) ||
        construct<OpenMPDeclarativeConstruct>(
            Parser<OpenMPDeclareTargetConstruct>{}) ||
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
TYPE_PARSER(maybe(startOmpLine >> "SECTION"_tok / endOmpLine) >>
    construct<OmpSectionBlocks>(
        nonemptySeparated(block, startOmpLine >> "SECTION"_tok / endOmpLine)))

// OMP SECTIONS (2.7.2), PARALLEL SECTIONS (2.11.2)
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
