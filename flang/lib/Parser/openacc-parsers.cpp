//===-- lib/Parser/openacc-parsers.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Top-level grammar specification for OpenACC 3.0.

#include "basic-parsers.h"
#include "expr-parsers.h"
#include "misc-parsers.h"
#include "stmt-parser.h"
#include "token-parsers.h"
#include "type-parser-implementation.h"
#include "flang/Parser/parse-tree.h"

// OpenACC Directives and Clauses
namespace Fortran::parser {

constexpr auto startAccLine = skipStuffBeforeStatement >> "!$ACC "_sptok;
constexpr auto endAccLine = space >> endOfLine;

// Basic clauses
TYPE_PARSER("AUTO" >> construct<AccClause>(construct<AccClause::Auto>()) ||
    "ASYNC" >> construct<AccClause>(construct<AccClause::Async>(
                   maybe(parenthesized(scalarIntExpr)))) ||
    "ATTACH" >> construct<AccClause>(construct<AccClause::Attach>(
                    parenthesized(Parser<AccObjectList>{}))) ||
    "BIND" >>
        construct<AccClause>(construct<AccClause::Bind>(parenthesized(name))) ||
    "CAPTURE" >> construct<AccClause>(construct<AccClause::Capture>()) ||
    "COLLAPSE" >> construct<AccClause>(construct<AccClause::Collapse>(
                      parenthesized(scalarIntConstantExpr))) ||
    ("COPY"_tok || "PRESENT_OR_COPY"_tok || "PCOPY"_tok) >>
        construct<AccClause>(construct<AccClause::Copy>(
            parenthesized(Parser<AccObjectList>{}))) ||
    ("COPYIN"_tok || "PRESENT_OR_COPYIN"_tok || "PCOPYIN"_tok) >>
        construct<AccClause>(construct<AccClause::Copyin>(
            parenthesized(Parser<AccObjectListWithModifier>{}))) ||
    ("COPYOUT"_tok || "PRESENT_OR_COPYOUT"_tok || "PCOPYOUT"_tok) >>
        construct<AccClause>(construct<AccClause::Copyout>(
            parenthesized(Parser<AccObjectListWithModifier>{}))) ||
    ("CREATE"_tok || "PRESENT_OR_CREATE"_tok || "PCREATE"_tok) >>
        construct<AccClause>(construct<AccClause::Create>(
            parenthesized(Parser<AccObjectListWithModifier>{}))) ||
    "DEFAULT" >> construct<AccClause>(construct<AccClause::Default>(
                     Parser<AccDefaultClause>{})) ||
    "DEFAULT_ASYNC" >> construct<AccClause>(construct<AccClause::DefaultAsync>(
                           parenthesized(scalarIntExpr))) ||
    "DELETE" >> construct<AccClause>(construct<AccClause::Delete>(
                    parenthesized(Parser<AccObjectList>{}))) ||
    "DETACH" >> construct<AccClause>(construct<AccClause::Detach>(
                    parenthesized(Parser<AccObjectList>{}))) ||
    "DEVICE" >> construct<AccClause>(construct<AccClause::Device>(
                    parenthesized(Parser<AccObjectList>{}))) ||
    "DEVICEPTR" >> construct<AccClause>(construct<AccClause::Deviceptr>(
                       parenthesized(Parser<AccObjectList>{}))) ||
    "DEVICE_NUM" >> construct<AccClause>(construct<AccClause::DeviceNum>(
                        parenthesized(scalarIntExpr))) ||
    "DEVICE_RESIDENT" >>
        construct<AccClause>(construct<AccClause::DeviceResident>(
            parenthesized(Parser<AccObjectList>{}))) ||
    ("DEVICE_TYPE"_tok || "DTYPE"_tok) >>
        construct<AccClause>(construct<AccClause::DeviceType>(parenthesized(
            "*" >> construct<std::optional<std::list<ScalarIntExpr>>>()))) ||
    ("DEVICE_TYPE"_tok || "DTYPE"_tok) >>
        construct<AccClause>(construct<AccClause::DeviceType>(
            parenthesized(maybe(nonemptyList(scalarIntExpr))))) ||
    "FINALIZE" >> construct<AccClause>(construct<AccClause::Finalize>()) ||
    "FIRSTPRIVATE" >> construct<AccClause>(construct<AccClause::Firstprivate>(
                          parenthesized(Parser<AccObjectList>{}))) ||
    "GANG" >> construct<AccClause>(construct<AccClause::Gang>(
                  maybe(parenthesized(Parser<AccGangArgument>{})))) ||
    "HOST" >> construct<AccClause>(construct<AccClause::Host>(
                  parenthesized(Parser<AccObjectList>{}))) ||
    "IF" >> construct<AccClause>(
                construct<AccClause::If>(parenthesized(scalarLogicalExpr))) ||
    "IF_PRESENT" >> construct<AccClause>(construct<AccClause::IfPresent>()) ||
    "INDEPENDENT" >>
        construct<AccClause>(construct<AccClause::Independent>()) ||
    "LINK" >> construct<AccClause>(construct<AccClause::Link>(
                  parenthesized(Parser<AccObjectList>{}))) ||
    "NO_CREATE" >> construct<AccClause>(construct<AccClause::NoCreate>(
                       parenthesized(Parser<AccObjectList>{}))) ||
    "NOHOST" >> construct<AccClause>(construct<AccClause::Nohost>()) ||
    "NUM_GANGS" >> construct<AccClause>(construct<AccClause::NumGangs>(
                       parenthesized(scalarIntExpr))) ||
    "NUM_WORKERS" >> construct<AccClause>(construct<AccClause::NumWorkers>(
                         parenthesized(scalarIntExpr))) ||
    "PRESENT" >> construct<AccClause>(construct<AccClause::Present>(
                     parenthesized(Parser<AccObjectList>{}))) ||
    "PRIVATE" >> construct<AccClause>(construct<AccClause::Private>(
                     parenthesized(Parser<AccObjectList>{}))) ||
    "READ" >> construct<AccClause>(construct<AccClause::Read>()) ||
    "REDUCTION" >> construct<AccClause>(construct<AccClause::Reduction>(
                       parenthesized(construct<AccObjectListWithReduction>(
                           Parser<AccReductionOperator>{} / ":",
                           Parser<AccObjectList>{})))) ||
    // SELF clause is either a simple optional condition for compute construct
    // or a synonym of the HOST clause for the update directive 2.14.4 holding
    // an object list.
    "SELF" >> construct<AccClause>(construct<AccClause::Self>(
                  maybe(parenthesized(scalarLogicalExpr)))) ||
    construct<AccClause>(
        construct<AccClause::Host>(parenthesized(Parser<AccObjectList>{}))) ||
    "SEQ" >> construct<AccClause>(construct<AccClause::Seq>()) ||
    "TILE" >> construct<AccClause>(construct<AccClause::Tile>(
                  parenthesized(Parser<AccTileExprList>{}))) ||
    "USE_DEVICE" >> construct<AccClause>(construct<AccClause::UseDevice>(
                        parenthesized(Parser<AccObjectList>{}))) ||
    "VECTOR_LENGTH" >> construct<AccClause>(construct<AccClause::VectorLength>(
                           parenthesized(scalarIntExpr))) ||
    "VECTOR" >>
        construct<AccClause>(construct<AccClause::Vector>(maybe(
            parenthesized(("LENGTH:" >> scalarIntExpr || scalarIntExpr))))) ||
    "WAIT" >> construct<AccClause>(construct<AccClause::Wait>(
                  maybe(parenthesized(Parser<AccWaitArgument>{})))) ||
    "WORKER" >>
        construct<AccClause>(construct<AccClause::Worker>(maybe(
            parenthesized(("NUM:" >> scalarIntExpr || scalarIntExpr))))) ||
    "WRITE" >> construct<AccClause>(construct<AccClause::Auto>()))

TYPE_PARSER(
    construct<AccObject>(designator) || construct<AccObject>("/" >> name / "/"))

TYPE_PARSER(construct<AccObjectList>(nonemptyList(Parser<AccObject>{})))

TYPE_PARSER(construct<AccObjectListWithModifier>(
    maybe(Parser<AccDataModifier>{}), Parser<AccObjectList>{}))

// 2.16.3 (2485) wait-argument is:
//   [devnum : int-expr :] [queues :] int-expr-list
TYPE_PARSER(construct<AccWaitArgument>(maybe("DEVNUM:" >> scalarIntExpr / ":"),
    "QUEUES:" >> nonemptyList(scalarIntExpr) || nonemptyList(scalarIntExpr)))

// 2.9 (1609) size-expr is one of:
//   * (represented as an empty std::optional<ScalarIntExpr>)
//   int-expr
TYPE_PARSER(construct<AccSizeExpr>(scalarIntExpr) ||
    construct<AccSizeExpr>("*" >> construct<std::optional<ScalarIntExpr>>()))
TYPE_PARSER(construct<AccSizeExprList>(nonemptyList(Parser<AccSizeExpr>{})))

// tile size is one of:
//   * (represented as an empty std::optional<ScalarIntExpr>)
//   constant-int-expr
TYPE_PARSER(construct<AccTileExpr>(scalarIntConstantExpr) ||
    construct<AccTileExpr>(
        "*" >> construct<std::optional<ScalarIntConstantExpr>>()))
TYPE_PARSER(construct<AccTileExprList>(nonemptyList(Parser<AccTileExpr>{})))

// 2.9 (1607) gang-arg is:
//   [[num:]int-expr][[,]static:size-expr]
TYPE_PARSER(construct<AccGangArgument>(
    maybe(("NUM:"_tok >> scalarIntExpr || scalarIntExpr)),
    maybe(", STATIC:" >> Parser<AccSizeExpr>{})))

// 2.5.13 Reduction
// Operator for reduction
TYPE_PARSER(sourced(construct<AccReductionOperator>(
    first("+" >> pure(AccReductionOperator::Operator::Plus),
        "*" >> pure(AccReductionOperator::Operator::Multiply),
        "MAX" >> pure(AccReductionOperator::Operator::Max),
        "MIN" >> pure(AccReductionOperator::Operator::Min),
        "IAND" >> pure(AccReductionOperator::Operator::Iand),
        "IOR" >> pure(AccReductionOperator::Operator::Ior),
        "IEOR" >> pure(AccReductionOperator::Operator::Ieor),
        ".AND." >> pure(AccReductionOperator::Operator::And),
        ".OR." >> pure(AccReductionOperator::Operator::Or),
        ".EQV." >> pure(AccReductionOperator::Operator::Eqv),
        ".NEQV." >> pure(AccReductionOperator::Operator::Neqv)))))

// 2.5.14 Default clause
TYPE_PARSER(construct<AccDefaultClause>(
    parenthesized(first("NONE" >> pure(AccDefaultClause::Arg::None),
        "PRESENT" >> pure(AccDefaultClause::Arg::Present)))))

// Modifier for copyin, copyout, cache and create
TYPE_PARSER(construct<AccDataModifier>(
    first("ZERO:" >> pure(AccDataModifier::Modifier::Zero),
        "READONLY:" >> pure(AccDataModifier::Modifier::ReadOnly))))

// Combined directives
TYPE_PARSER(sourced(construct<AccCombinedDirective>(
    first("KERNELS LOOP" >> pure(llvm::acc::Directive::ACCD_kernels_loop),
        "PARALLEL LOOP" >> pure(llvm::acc::Directive::ACCD_parallel_loop),
        "SERIAL LOOP" >> pure(llvm::acc::Directive::ACCD_serial_loop)))))

// Block directives
TYPE_PARSER(sourced(construct<AccBlockDirective>(
    first("DATA" >> pure(llvm::acc::Directive::ACCD_data),
        "HOST_DATA" >> pure(llvm::acc::Directive::ACCD_host_data),
        "KERNELS" >> pure(llvm::acc::Directive::ACCD_kernels),
        "PARALLEL" >> pure(llvm::acc::Directive::ACCD_parallel),
        "SERIAL" >> pure(llvm::acc::Directive::ACCD_serial)))))

// Standalone directives
TYPE_PARSER(sourced(construct<AccStandaloneDirective>(
    first("ENTER DATA" >> pure(llvm::acc::Directive::ACCD_enter_data),
        "EXIT DATA" >> pure(llvm::acc::Directive::ACCD_exit_data),
        "INIT" >> pure(llvm::acc::Directive::ACCD_init),
        "SHUTDOWN" >> pure(llvm::acc::Directive::ACCD_shutdown),
        "SET" >> pure(llvm::acc::Directive::ACCD_set),
        "UPDATE" >> pure(llvm::acc::Directive::ACCD_update)))))

// Loop directives
TYPE_PARSER(sourced(construct<AccLoopDirective>(
    first("LOOP" >> pure(llvm::acc::Directive::ACCD_loop)))))

TYPE_PARSER(construct<AccBeginLoopDirective>(
    sourced(Parser<AccLoopDirective>{}), Parser<AccClauseList>{}))

TYPE_PARSER(
    construct<OpenACCLoopConstruct>(sourced(Parser<AccBeginLoopDirective>{})))

// 2.15.1 Routine directive
TYPE_PARSER(sourced(construct<OpenACCRoutineConstruct>(verbatim("ROUTINE"_tok),
    maybe(parenthesized(name)), Parser<AccClauseList>{})))

// 2.10 Cache directive
TYPE_PARSER(sourced(
    construct<OpenACCCacheConstruct>(sourced(construct<Verbatim>("CACHE"_tok)),
        parenthesized(Parser<AccObjectListWithModifier>{}))))

// 2.11 Combined constructs
TYPE_PARSER(construct<AccBeginCombinedDirective>(
    sourced(Parser<AccCombinedDirective>{}), Parser<AccClauseList>{}))

// 2.12 Atomic constructs
TYPE_PARSER(construct<AccEndAtomic>(startAccLine >> "END ATOMIC"_tok))

TYPE_PARSER("ATOMIC" >>
    construct<AccAtomicRead>(verbatim("READ"_tok) / endAccLine,
        statement(assignmentStmt), maybe(Parser<AccEndAtomic>{} / endAccLine)))

TYPE_PARSER("ATOMIC" >>
    construct<AccAtomicWrite>(verbatim("WRITE"_tok) / endAccLine,
        statement(assignmentStmt), maybe(Parser<AccEndAtomic>{} / endAccLine)))

TYPE_PARSER("ATOMIC" >>
    construct<AccAtomicUpdate>(maybe(verbatim("UPDATE"_tok)) / endAccLine,
        statement(assignmentStmt), maybe(Parser<AccEndAtomic>{} / endAccLine)))

TYPE_PARSER("ATOMIC" >>
    construct<AccAtomicCapture>(verbatim("CAPTURE"_tok) / endAccLine,
        statement(assignmentStmt), statement(assignmentStmt),
        Parser<AccEndAtomic>{} / endAccLine))

TYPE_PARSER(
    sourced(construct<OpenACCAtomicConstruct>(Parser<AccAtomicRead>{})) ||
    sourced(construct<OpenACCAtomicConstruct>(Parser<AccAtomicCapture>{})) ||
    sourced(construct<OpenACCAtomicConstruct>(Parser<AccAtomicWrite>{})) ||
    sourced(construct<OpenACCAtomicConstruct>(Parser<AccAtomicUpdate>{})))

// 2.13 Declare constructs
TYPE_PARSER(sourced(construct<AccDeclarativeDirective>(
    first("DECLARE" >> pure(llvm::acc::Directive::ACCD_declare)))))

// [Clause, [Clause], ...]
TYPE_PARSER(sourced(construct<AccClauseList>(
    many(maybe(","_tok) >> sourced(Parser<AccClause>{})))))

// 2.16.3 Wait directive
TYPE_PARSER(sourced(construct<OpenACCWaitConstruct>(
    sourced(construct<Verbatim>("WAIT"_tok)),
    maybe(parenthesized(Parser<AccWaitArgument>{})), Parser<AccClauseList>{})))

// Block Constructs
TYPE_PARSER(sourced(construct<AccBeginBlockDirective>(
    sourced(Parser<AccBlockDirective>{}), Parser<AccClauseList>{})))

TYPE_PARSER(startAccLine >> sourced(construct<AccEndBlockDirective>("END"_tok >>
                                sourced(Parser<AccBlockDirective>{}))))

TYPE_PARSER(construct<OpenACCBlockConstruct>(
    Parser<AccBeginBlockDirective>{} / endAccLine, block,
    Parser<AccEndBlockDirective>{} / endAccLine))

// Standalone constructs
TYPE_PARSER(construct<OpenACCStandaloneConstruct>(
    sourced(Parser<AccStandaloneDirective>{}), Parser<AccClauseList>{}))

// Standalone declarative constructs
TYPE_PARSER(construct<OpenACCStandaloneDeclarativeConstruct>(
    sourced(Parser<AccDeclarativeDirective>{}), Parser<AccClauseList>{}))

TYPE_PARSER(
    startAccLine >> sourced(construct<OpenACCDeclarativeConstruct>(
                        Parser<OpenACCStandaloneDeclarativeConstruct>{})))

// OpenACC constructs
TYPE_CONTEXT_PARSER("OpenACC construct"_en_US,
    startAccLine >>
        first(construct<OpenACCConstruct>(Parser<OpenACCBlockConstruct>{}),
            construct<OpenACCConstruct>(Parser<OpenACCCombinedConstruct>{}),
            construct<OpenACCConstruct>(Parser<OpenACCLoopConstruct>{}),
            construct<OpenACCConstruct>(Parser<OpenACCStandaloneConstruct>{}),
            construct<OpenACCConstruct>(Parser<OpenACCRoutineConstruct>{}),
            construct<OpenACCConstruct>(Parser<OpenACCCacheConstruct>{}),
            construct<OpenACCConstruct>(Parser<OpenACCWaitConstruct>{}),
            construct<OpenACCConstruct>(Parser<OpenACCAtomicConstruct>{})))

TYPE_PARSER(startAccLine >> sourced(construct<AccEndCombinedDirective>(sourced(
                                "END"_tok >> Parser<AccCombinedDirective>{}))))

TYPE_PARSER(construct<OpenACCCombinedConstruct>(
    sourced(Parser<AccBeginCombinedDirective>{} / endAccLine)))

} // namespace Fortran::parser
