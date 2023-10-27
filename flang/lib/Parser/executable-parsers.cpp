//===-- lib/Parser/executable-parsers.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Per-type parsers for executable statements

#include "basic-parsers.h"
#include "debug-parser.h"
#include "expr-parsers.h"
#include "misc-parsers.h"
#include "stmt-parser.h"
#include "token-parsers.h"
#include "type-parser-implementation.h"
#include "flang/Parser/characters.h"
#include "flang/Parser/parse-tree.h"

namespace Fortran::parser {

// Fortran allows the statement with the corresponding label at the end of
// a do-construct that begins with an old-style label-do-stmt to be a
// new-style END DO statement; e.g., DO 10 I=1,N; ...; 10 END DO.  Usually,
// END DO statements appear only at the ends of do-constructs that begin
// with a nonlabel-do-stmt, so care must be taken to recognize this case and
// essentially treat them like CONTINUE statements.

// R514 executable-construct ->
//        action-stmt | associate-construct | block-construct |
//        case-construct | change-team-construct | critical-construct |
//        do-construct | if-construct | select-rank-construct |
//        select-type-construct | where-construct | forall-construct |
// (CUDA) CUF-kernel-do-construct
constexpr auto executableConstruct{first(
    construct<ExecutableConstruct>(CapturedLabelDoStmt{}),
    construct<ExecutableConstruct>(EndDoStmtForCapturedLabelDoStmt{}),
    construct<ExecutableConstruct>(indirect(Parser<DoConstruct>{})),
    // Attempt DO statements before assignment statements for better
    // error messages in cases like "DO10I=1,(error)".
    construct<ExecutableConstruct>(statement(actionStmt)),
    construct<ExecutableConstruct>(indirect(Parser<AssociateConstruct>{})),
    construct<ExecutableConstruct>(indirect(Parser<BlockConstruct>{})),
    construct<ExecutableConstruct>(indirect(Parser<CaseConstruct>{})),
    construct<ExecutableConstruct>(indirect(Parser<ChangeTeamConstruct>{})),
    construct<ExecutableConstruct>(indirect(Parser<CriticalConstruct>{})),
    construct<ExecutableConstruct>(indirect(Parser<IfConstruct>{})),
    construct<ExecutableConstruct>(indirect(Parser<SelectRankConstruct>{})),
    construct<ExecutableConstruct>(indirect(Parser<SelectTypeConstruct>{})),
    construct<ExecutableConstruct>(indirect(whereConstruct)),
    construct<ExecutableConstruct>(indirect(forallConstruct)),
    construct<ExecutableConstruct>(indirect(ompEndLoopDirective)),
    construct<ExecutableConstruct>(indirect(openmpConstruct)),
    construct<ExecutableConstruct>(indirect(Parser<OpenACCConstruct>{})),
    construct<ExecutableConstruct>(indirect(compilerDirective)),
    construct<ExecutableConstruct>(indirect(Parser<CUFKernelDoConstruct>{})))};

// R510 execution-part-construct ->
//        executable-construct | format-stmt | entry-stmt | data-stmt
// Extension (PGI/Intel): also accept NAMELIST in execution part
constexpr auto obsoleteExecutionPartConstruct{recovery(ignoredStatementPrefix >>
        fail<ExecutionPartConstruct>(
            "obsolete legacy extension is not supported"_err_en_US),
    construct<ExecutionPartConstruct>(construct<ErrorRecovery>(ok /
        statement("REDIMENSION" >> name /
                parenthesized(nonemptyList(Parser<AllocateShapeSpec>{}))))))};

TYPE_PARSER(recovery(
    withMessage("expected execution part construct"_err_en_US,
        CONTEXT_PARSER("execution part construct"_en_US,
            first(construct<ExecutionPartConstruct>(executableConstruct),
                construct<ExecutionPartConstruct>(
                    statement(indirect(formatStmt))),
                construct<ExecutionPartConstruct>(
                    statement(indirect(entryStmt))),
                construct<ExecutionPartConstruct>(
                    statement(indirect(dataStmt))),
                extension<LanguageFeature::ExecutionPartNamelist>(
                    "nonstandard usage: NAMELIST in execution part"_port_en_US,
                    construct<ExecutionPartConstruct>(
                        statement(indirect(Parser<NamelistStmt>{})))),
                obsoleteExecutionPartConstruct))),
    construct<ExecutionPartConstruct>(executionPartErrorRecovery)))

// R509 execution-part -> executable-construct [execution-part-construct]...
TYPE_CONTEXT_PARSER("execution part"_en_US,
    construct<ExecutionPart>(many(executionPartConstruct)))

// R515 action-stmt ->
//        allocate-stmt | assignment-stmt | backspace-stmt | call-stmt |
//        close-stmt | continue-stmt | cycle-stmt | deallocate-stmt |
//        endfile-stmt | error-stop-stmt | event-post-stmt | event-wait-stmt |
//        exit-stmt | fail-image-stmt | flush-stmt | form-team-stmt |
//        goto-stmt | if-stmt | inquire-stmt | lock-stmt | nullify-stmt |
//        open-stmt | pointer-assignment-stmt | print-stmt | read-stmt |
//        return-stmt | rewind-stmt | stop-stmt | sync-all-stmt |
//        sync-images-stmt | sync-memory-stmt | sync-team-stmt | unlock-stmt |
//        wait-stmt | where-stmt | write-stmt | computed-goto-stmt | forall-stmt
// R1159 continue-stmt -> CONTINUE
// R1163 fail-image-stmt -> FAIL IMAGE
TYPE_PARSER(first(construct<ActionStmt>(indirect(Parser<AllocateStmt>{})),
    construct<ActionStmt>(indirect(assignmentStmt)),
    construct<ActionStmt>(indirect(pointerAssignmentStmt)),
    construct<ActionStmt>(indirect(Parser<BackspaceStmt>{})),
    construct<ActionStmt>(indirect(Parser<CallStmt>{})),
    construct<ActionStmt>(indirect(Parser<CloseStmt>{})),
    construct<ActionStmt>(construct<ContinueStmt>("CONTINUE"_tok)),
    construct<ActionStmt>(indirect(Parser<CycleStmt>{})),
    construct<ActionStmt>(indirect(Parser<DeallocateStmt>{})),
    construct<ActionStmt>(indirect(Parser<EndfileStmt>{})),
    construct<ActionStmt>(indirect(Parser<EventPostStmt>{})),
    construct<ActionStmt>(indirect(Parser<EventWaitStmt>{})),
    construct<ActionStmt>(indirect(Parser<ExitStmt>{})),
    construct<ActionStmt>(construct<FailImageStmt>("FAIL IMAGE"_sptok)),
    construct<ActionStmt>(indirect(Parser<FlushStmt>{})),
    construct<ActionStmt>(indirect(Parser<FormTeamStmt>{})),
    construct<ActionStmt>(indirect(Parser<GotoStmt>{})),
    construct<ActionStmt>(indirect(Parser<IfStmt>{})),
    construct<ActionStmt>(indirect(Parser<InquireStmt>{})),
    construct<ActionStmt>(indirect(Parser<LockStmt>{})),
    construct<ActionStmt>(indirect(Parser<NullifyStmt>{})),
    construct<ActionStmt>(indirect(Parser<OpenStmt>{})),
    construct<ActionStmt>(indirect(Parser<PrintStmt>{})),
    construct<ActionStmt>(indirect(Parser<ReadStmt>{})),
    construct<ActionStmt>(indirect(Parser<ReturnStmt>{})),
    construct<ActionStmt>(indirect(Parser<RewindStmt>{})),
    construct<ActionStmt>(indirect(Parser<StopStmt>{})), // & error-stop-stmt
    construct<ActionStmt>(indirect(Parser<SyncAllStmt>{})),
    construct<ActionStmt>(indirect(Parser<SyncImagesStmt>{})),
    construct<ActionStmt>(indirect(Parser<SyncMemoryStmt>{})),
    construct<ActionStmt>(indirect(Parser<SyncTeamStmt>{})),
    construct<ActionStmt>(indirect(Parser<UnlockStmt>{})),
    construct<ActionStmt>(indirect(Parser<WaitStmt>{})),
    construct<ActionStmt>(indirect(whereStmt)),
    construct<ActionStmt>(indirect(Parser<WriteStmt>{})),
    construct<ActionStmt>(indirect(Parser<ComputedGotoStmt>{})),
    construct<ActionStmt>(indirect(forallStmt)),
    construct<ActionStmt>(indirect(Parser<ArithmeticIfStmt>{})),
    construct<ActionStmt>(indirect(Parser<AssignStmt>{})),
    construct<ActionStmt>(indirect(Parser<AssignedGotoStmt>{})),
    construct<ActionStmt>(indirect(Parser<PauseStmt>{}))))

// R1102 associate-construct -> associate-stmt block end-associate-stmt
TYPE_CONTEXT_PARSER("ASSOCIATE construct"_en_US,
    construct<AssociateConstruct>(statement(Parser<AssociateStmt>{}), block,
        statement(Parser<EndAssociateStmt>{})))

// R1103 associate-stmt ->
//        [associate-construct-name :] ASSOCIATE ( association-list )
TYPE_CONTEXT_PARSER("ASSOCIATE statement"_en_US,
    construct<AssociateStmt>(maybe(name / ":"),
        "ASSOCIATE" >> parenthesized(nonemptyList(Parser<Association>{}))))

// R1104 association -> associate-name => selector
TYPE_PARSER(construct<Association>(name, "=>" >> selector))

// R1105 selector -> expr | variable
TYPE_PARSER(construct<Selector>(variable) / lookAhead(","_tok || ")"_tok) ||
    construct<Selector>(expr))

// R1106 end-associate-stmt -> END ASSOCIATE [associate-construct-name]
TYPE_PARSER(construct<EndAssociateStmt>(recovery(
    "END ASSOCIATE" >> maybe(name), namedConstructEndStmtErrorRecovery)))

// R1107 block-construct ->
//         block-stmt [block-specification-part] block end-block-stmt
TYPE_CONTEXT_PARSER("BLOCK construct"_en_US,
    construct<BlockConstruct>(statement(Parser<BlockStmt>{}),
        Parser<BlockSpecificationPart>{}, // can be empty
        block, statement(Parser<EndBlockStmt>{})))

// R1108 block-stmt -> [block-construct-name :] BLOCK
TYPE_PARSER(construct<BlockStmt>(maybe(name / ":") / "BLOCK"))

// R1109 block-specification-part ->
//         [use-stmt]... [import-stmt]... [implicit-part]
//         [[declaration-construct]... specification-construct]
// C1107 prohibits COMMON, EQUIVALENCE, INTENT, NAMELIST, OPTIONAL, VALUE,
// and statement function definitions.  C1108 prohibits SAVE /common/.
// C1570 indirectly prohibits ENTRY.  These constraints are best enforced later.
// The odd grammar rule above would have the effect of forcing any
// trailing FORMAT and DATA statements after the last specification-construct
// to be recognized as part of the block-construct's block part rather than
// its block-specification-part, a distinction without any apparent difference.
TYPE_PARSER(construct<BlockSpecificationPart>(specificationPart))

// R1110 end-block-stmt -> END BLOCK [block-construct-name]
TYPE_PARSER(construct<EndBlockStmt>(
    recovery("END BLOCK" >> maybe(name), namedConstructEndStmtErrorRecovery)))

// R1111 change-team-construct -> change-team-stmt block end-change-team-stmt
TYPE_CONTEXT_PARSER("CHANGE TEAM construct"_en_US,
    construct<ChangeTeamConstruct>(statement(Parser<ChangeTeamStmt>{}), block,
        statement(Parser<EndChangeTeamStmt>{})))

// R1112 change-team-stmt ->
//         [team-construct-name :] CHANGE TEAM
//         ( team-value [, coarray-association-list] [, sync-stat-list] )
TYPE_CONTEXT_PARSER("CHANGE TEAM statement"_en_US,
    construct<ChangeTeamStmt>(maybe(name / ":"),
        "CHANGE TEAM"_sptok >> "("_tok >> teamValue,
        defaulted("," >> nonemptyList(Parser<CoarrayAssociation>{})),
        defaulted("," >> nonemptyList(statOrErrmsg))) /
        ")")

// R1113 coarray-association -> codimension-decl => selector
TYPE_PARSER(
    construct<CoarrayAssociation>(Parser<CodimensionDecl>{}, "=>" >> selector))

// R1114 end-change-team-stmt ->
//         END TEAM [( [sync-stat-list] )] [team-construct-name]
TYPE_CONTEXT_PARSER("END TEAM statement"_en_US,
    construct<EndChangeTeamStmt>(
        "END TEAM" >> defaulted(parenthesized(optionalList(statOrErrmsg))),
        maybe(name)))

// R1117 critical-stmt ->
//         [critical-construct-name :] CRITICAL [( [sync-stat-list] )]
TYPE_CONTEXT_PARSER("CRITICAL statement"_en_US,
    construct<CriticalStmt>(maybe(name / ":"),
        "CRITICAL" >> defaulted(parenthesized(optionalList(statOrErrmsg)))))

// R1116 critical-construct -> critical-stmt block end-critical-stmt
TYPE_CONTEXT_PARSER("CRITICAL construct"_en_US,
    construct<CriticalConstruct>(statement(Parser<CriticalStmt>{}), block,
        statement(Parser<EndCriticalStmt>{})))

// R1118 end-critical-stmt -> END CRITICAL [critical-construct-name]
TYPE_PARSER(construct<EndCriticalStmt>(recovery(
    "END CRITICAL" >> maybe(name), namedConstructEndStmtErrorRecovery)))

// R1119 do-construct -> do-stmt block end-do
// R1120 do-stmt -> nonlabel-do-stmt | label-do-stmt
TYPE_CONTEXT_PARSER("DO construct"_en_US,
    construct<DoConstruct>(
        statement(Parser<NonLabelDoStmt>{}) / EnterNonlabelDoConstruct{}, block,
        statement(Parser<EndDoStmt>{}) / LeaveDoConstruct{}))

// R1125 concurrent-header ->
//         ( [integer-type-spec ::] concurrent-control-list
//         [, scalar-mask-expr] )
TYPE_PARSER(parenthesized(construct<ConcurrentHeader>(
    maybe(integerTypeSpec / "::"), nonemptyList(Parser<ConcurrentControl>{}),
    maybe("," >> scalarLogicalExpr))))

// R1126 concurrent-control ->
//         index-name = concurrent-limit : concurrent-limit [: concurrent-step]
// R1127 concurrent-limit -> scalar-int-expr
// R1128 concurrent-step -> scalar-int-expr
TYPE_PARSER(construct<ConcurrentControl>(name / "=", scalarIntExpr / ":",
    scalarIntExpr, maybe(":" >> scalarIntExpr)))

// R1130 locality-spec ->
//         LOCAL ( variable-name-list ) | LOCAL_INIT ( variable-name-list ) |
//         SHARED ( variable-name-list ) | DEFAULT ( NONE )
TYPE_PARSER(construct<LocalitySpec>(construct<LocalitySpec::Local>(
                "LOCAL" >> parenthesized(listOfNames))) ||
    construct<LocalitySpec>(construct<LocalitySpec::LocalInit>(
        "LOCAL_INIT"_sptok >> parenthesized(listOfNames))) ||
    construct<LocalitySpec>(construct<LocalitySpec::Shared>(
        "SHARED" >> parenthesized(listOfNames))) ||
    construct<LocalitySpec>(
        construct<LocalitySpec::DefaultNone>("DEFAULT ( NONE )"_tok)))

// R1123 loop-control ->
//         [,] do-variable = scalar-int-expr , scalar-int-expr
//           [, scalar-int-expr] |
//         [,] WHILE ( scalar-logical-expr ) |
//         [,] CONCURRENT concurrent-header concurrent-locality
// R1129 concurrent-locality -> [locality-spec]...
TYPE_CONTEXT_PARSER("loop control"_en_US,
    maybe(","_tok) >>
        (construct<LoopControl>(loopBounds(scalarExpr)) ||
            construct<LoopControl>(
                "WHILE" >> parenthesized(scalarLogicalExpr)) ||
            construct<LoopControl>(construct<LoopControl::Concurrent>(
                "CONCURRENT" >> concurrentHeader,
                many(Parser<LocalitySpec>{})))))

// R1121 label-do-stmt -> [do-construct-name :] DO label [loop-control]
TYPE_CONTEXT_PARSER("label DO statement"_en_US,
    construct<LabelDoStmt>(
        maybe(name / ":"), "DO" >> label, maybe(loopControl)))

// R1122 nonlabel-do-stmt -> [do-construct-name :] DO [loop-control]
TYPE_CONTEXT_PARSER("nonlabel DO statement"_en_US,
    construct<NonLabelDoStmt>(maybe(name / ":"), "DO" >> maybe(loopControl)))

// R1132 end-do-stmt -> END DO [do-construct-name]
TYPE_CONTEXT_PARSER("END DO statement"_en_US,
    construct<EndDoStmt>(
        recovery("END DO" >> maybe(name), namedConstructEndStmtErrorRecovery)))

// R1133 cycle-stmt -> CYCLE [do-construct-name]
TYPE_CONTEXT_PARSER(
    "CYCLE statement"_en_US, construct<CycleStmt>("CYCLE" >> maybe(name)))

// R1134 if-construct ->
//         if-then-stmt block [else-if-stmt block]...
//         [else-stmt block] end-if-stmt
// R1135 if-then-stmt -> [if-construct-name :] IF ( scalar-logical-expr )
// THEN R1136 else-if-stmt ->
//         ELSE IF ( scalar-logical-expr ) THEN [if-construct-name]
// R1137 else-stmt -> ELSE [if-construct-name]
// R1138 end-if-stmt -> END IF [if-construct-name]
TYPE_CONTEXT_PARSER("IF construct"_en_US,
    construct<IfConstruct>(
        statement(construct<IfThenStmt>(maybe(name / ":"),
            "IF" >> parenthesized(scalarLogicalExpr) /
                    recovery("THEN"_tok, lookAhead(endOfStmt)))),
        block,
        many(construct<IfConstruct::ElseIfBlock>(
            unambiguousStatement(construct<ElseIfStmt>(
                "ELSE IF" >> parenthesized(scalarLogicalExpr),
                recovery("THEN"_tok, ok) >> maybe(name))),
            block)),
        maybe(construct<IfConstruct::ElseBlock>(
            statement(construct<ElseStmt>("ELSE" >> maybe(name))), block)),
        statement(construct<EndIfStmt>(recovery(
            "END IF" >> maybe(name), namedConstructEndStmtErrorRecovery)))))

// R1139 if-stmt -> IF ( scalar-logical-expr ) action-stmt
TYPE_CONTEXT_PARSER("IF statement"_en_US,
    construct<IfStmt>("IF" >> parenthesized(scalarLogicalExpr),
        unlabeledStatement(actionStmt)))

// R1140 case-construct ->
//         select-case-stmt [case-stmt block]... end-select-stmt
TYPE_CONTEXT_PARSER("SELECT CASE construct"_en_US,
    construct<CaseConstruct>(statement(Parser<SelectCaseStmt>{}),
        many(construct<CaseConstruct::Case>(
            unambiguousStatement(Parser<CaseStmt>{}), block)),
        statement(endSelectStmt)))

// R1141 select-case-stmt -> [case-construct-name :] SELECT CASE ( case-expr
// ) R1144 case-expr -> scalar-expr
TYPE_CONTEXT_PARSER("SELECT CASE statement"_en_US,
    construct<SelectCaseStmt>(
        maybe(name / ":"), "SELECT CASE" >> parenthesized(scalar(expr))))

// R1142 case-stmt -> CASE case-selector [case-construct-name]
TYPE_CONTEXT_PARSER("CASE statement"_en_US,
    construct<CaseStmt>("CASE" >> Parser<CaseSelector>{}, maybe(name)))

// R1143 end-select-stmt -> END SELECT [case-construct-name]
// R1151 end-select-rank-stmt -> END SELECT [select-construct-name]
// R1155 end-select-type-stmt -> END SELECT [select-construct-name]
TYPE_PARSER(construct<EndSelectStmt>(
    recovery("END SELECT" >> maybe(name), namedConstructEndStmtErrorRecovery)))

// R1145 case-selector -> ( case-value-range-list ) | DEFAULT
constexpr auto defaultKeyword{construct<Default>("DEFAULT"_tok)};
TYPE_PARSER(parenthesized(construct<CaseSelector>(
                nonemptyList(Parser<CaseValueRange>{}))) ||
    construct<CaseSelector>(defaultKeyword))

// R1147 case-value -> scalar-constant-expr
constexpr auto caseValue{scalar(constantExpr)};

// R1146 case-value-range ->
//         case-value | case-value : | : case-value | case-value : case-value
TYPE_PARSER(construct<CaseValueRange>(construct<CaseValueRange::Range>(
                construct<std::optional<CaseValue>>(caseValue),
                ":" >> maybe(caseValue))) ||
    construct<CaseValueRange>(
        construct<CaseValueRange::Range>(construct<std::optional<CaseValue>>(),
            ":" >> construct<std::optional<CaseValue>>(caseValue))) ||
    construct<CaseValueRange>(caseValue))

// R1148 select-rank-construct ->
//         select-rank-stmt [select-rank-case-stmt block]...
//         end-select-rank-stmt
TYPE_CONTEXT_PARSER("SELECT RANK construct"_en_US,
    construct<SelectRankConstruct>(statement(Parser<SelectRankStmt>{}),
        many(construct<SelectRankConstruct::RankCase>(
            unambiguousStatement(Parser<SelectRankCaseStmt>{}), block)),
        statement(endSelectStmt)))

// R1149 select-rank-stmt ->
//         [select-construct-name :] SELECT RANK
//         ( [associate-name =>] selector )
TYPE_CONTEXT_PARSER("SELECT RANK statement"_en_US,
    construct<SelectRankStmt>(maybe(name / ":"),
        "SELECT RANK"_sptok >> "("_tok >> maybe(name / "=>"), selector / ")"))

// R1150 select-rank-case-stmt ->
//         RANK ( scalar-int-constant-expr ) [select-construct-name] |
//         RANK ( * ) [select-construct-name] |
//         RANK DEFAULT [select-construct-name]
TYPE_CONTEXT_PARSER("RANK case statement"_en_US,
    "RANK" >> (construct<SelectRankCaseStmt>(
                  parenthesized(construct<SelectRankCaseStmt::Rank>(
                                    scalarIntConstantExpr) ||
                      construct<SelectRankCaseStmt::Rank>(star)) ||
                      construct<SelectRankCaseStmt::Rank>(defaultKeyword),
                  maybe(name))))

// R1152 select-type-construct ->
//         select-type-stmt [type-guard-stmt block]... end-select-type-stmt
TYPE_CONTEXT_PARSER("SELECT TYPE construct"_en_US,
    construct<SelectTypeConstruct>(statement(Parser<SelectTypeStmt>{}),
        many(construct<SelectTypeConstruct::TypeCase>(
            unambiguousStatement(Parser<TypeGuardStmt>{}), block)),
        statement(endSelectStmt)))

// R1153 select-type-stmt ->
//         [select-construct-name :] SELECT TYPE
//         ( [associate-name =>] selector )
TYPE_CONTEXT_PARSER("SELECT TYPE statement"_en_US,
    construct<SelectTypeStmt>(maybe(name / ":"),
        "SELECT TYPE (" >> maybe(name / "=>"), selector / ")"))

// R1154 type-guard-stmt ->
//         TYPE IS ( type-spec ) [select-construct-name] |
//         CLASS IS ( derived-type-spec ) [select-construct-name] |
//         CLASS DEFAULT [select-construct-name]
TYPE_CONTEXT_PARSER("type guard statement"_en_US,
    construct<TypeGuardStmt>("TYPE IS"_sptok >>
                parenthesized(construct<TypeGuardStmt::Guard>(typeSpec)) ||
            "CLASS IS"_sptok >> parenthesized(construct<TypeGuardStmt::Guard>(
                                    derivedTypeSpec)) ||
            construct<TypeGuardStmt::Guard>("CLASS" >> defaultKeyword),
        maybe(name)))

// R1156 exit-stmt -> EXIT [construct-name]
TYPE_CONTEXT_PARSER(
    "EXIT statement"_en_US, construct<ExitStmt>("EXIT" >> maybe(name)))

// R1157 goto-stmt -> GO TO label
TYPE_CONTEXT_PARSER(
    "GOTO statement"_en_US, construct<GotoStmt>("GO TO" >> label))

// R1158 computed-goto-stmt -> GO TO ( label-list ) [,] scalar-int-expr
TYPE_CONTEXT_PARSER("computed GOTO statement"_en_US,
    construct<ComputedGotoStmt>("GO TO" >> parenthesized(nonemptyList(label)),
        maybe(","_tok) >> scalarIntExpr))

// R1160 stop-stmt -> STOP [stop-code] [, QUIET = scalar-logical-expr]
// R1161 error-stop-stmt ->
//         ERROR STOP [stop-code] [, QUIET = scalar-logical-expr]
TYPE_CONTEXT_PARSER("STOP statement"_en_US,
    construct<StopStmt>("STOP" >> pure(StopStmt::Kind::Stop) ||
            "ERROR STOP"_sptok >> pure(StopStmt::Kind::ErrorStop),
        maybe(Parser<StopCode>{}), maybe(", QUIET =" >> scalarLogicalExpr)))

// R1162 stop-code -> scalar-default-char-expr | scalar-int-expr
// The two alternatives for stop-code can't be distinguished at
// parse time.
TYPE_PARSER(construct<StopCode>(scalar(expr)))

// R1164 sync-all-stmt -> SYNC ALL [( [sync-stat-list] )]
TYPE_CONTEXT_PARSER("SYNC ALL statement"_en_US,
    construct<SyncAllStmt>("SYNC ALL"_sptok >>
        defaulted(parenthesized(optionalList(statOrErrmsg)))))

// R1166 sync-images-stmt -> SYNC IMAGES ( image-set [, sync-stat-list] )
// R1167 image-set -> int-expr | *
TYPE_CONTEXT_PARSER("SYNC IMAGES statement"_en_US,
    "SYNC IMAGES"_sptok >> parenthesized(construct<SyncImagesStmt>(
                               construct<SyncImagesStmt::ImageSet>(intExpr) ||
                                   construct<SyncImagesStmt::ImageSet>(star),
                               defaulted("," >> nonemptyList(statOrErrmsg)))))

// R1168 sync-memory-stmt -> SYNC MEMORY [( [sync-stat-list] )]
TYPE_CONTEXT_PARSER("SYNC MEMORY statement"_en_US,
    construct<SyncMemoryStmt>("SYNC MEMORY"_sptok >>
        defaulted(parenthesized(optionalList(statOrErrmsg)))))

// R1169 sync-team-stmt -> SYNC TEAM ( team-value [, sync-stat-list] )
TYPE_CONTEXT_PARSER("SYNC TEAM statement"_en_US,
    construct<SyncTeamStmt>("SYNC TEAM"_sptok >> "("_tok >> teamValue,
        defaulted("," >> nonemptyList(statOrErrmsg)) / ")"))

// R1170 event-post-stmt -> EVENT POST ( event-variable [, sync-stat-list] )
// R1171 event-variable -> scalar-variable
TYPE_CONTEXT_PARSER("EVENT POST statement"_en_US,
    construct<EventPostStmt>("EVENT POST"_sptok >> "("_tok >> scalar(variable),
        defaulted("," >> nonemptyList(statOrErrmsg)) / ")"))

// R1172 event-wait-stmt ->
//         EVENT WAIT ( event-variable [, event-wait-spec-list] )
TYPE_CONTEXT_PARSER("EVENT WAIT statement"_en_US,
    construct<EventWaitStmt>("EVENT WAIT"_sptok >> "("_tok >> scalar(variable),
        defaulted("," >> nonemptyList(Parser<EventWaitStmt::EventWaitSpec>{})) /
            ")"))

// R1174 until-spec -> UNTIL_COUNT = scalar-int-expr
constexpr auto untilSpec{"UNTIL_COUNT =" >> scalarIntExpr};

// R1173 event-wait-spec -> until-spec | sync-stat
TYPE_PARSER(construct<EventWaitStmt::EventWaitSpec>(untilSpec) ||
    construct<EventWaitStmt::EventWaitSpec>(statOrErrmsg))

// R1177 team-variable -> scalar-variable
constexpr auto teamVariable{scalar(variable)};

// R1175 form-team-stmt ->
//         FORM TEAM ( team-number , team-variable [, form-team-spec-list] )
// R1176 team-number -> scalar-int-expr
TYPE_CONTEXT_PARSER("FORM TEAM statement"_en_US,
    construct<FormTeamStmt>("FORM TEAM"_sptok >> "("_tok >> scalarIntExpr,
        "," >> teamVariable,
        defaulted("," >> nonemptyList(Parser<FormTeamStmt::FormTeamSpec>{})) /
            ")"))

// R1178 form-team-spec -> NEW_INDEX = scalar-int-expr | sync-stat
TYPE_PARSER(
    construct<FormTeamStmt::FormTeamSpec>("NEW_INDEX =" >> scalarIntExpr) ||
    construct<FormTeamStmt::FormTeamSpec>(statOrErrmsg))

// R1182 lock-variable -> scalar-variable
constexpr auto lockVariable{scalar(variable)};

// R1179 lock-stmt -> LOCK ( lock-variable [, lock-stat-list] )
TYPE_CONTEXT_PARSER("LOCK statement"_en_US,
    construct<LockStmt>("LOCK (" >> lockVariable,
        defaulted("," >> nonemptyList(Parser<LockStmt::LockStat>{})) / ")"))

// R1180 lock-stat -> ACQUIRED_LOCK = scalar-logical-variable | sync-stat
TYPE_PARSER(
    construct<LockStmt::LockStat>("ACQUIRED_LOCK =" >> scalarLogicalVariable) ||
    construct<LockStmt::LockStat>(statOrErrmsg))

// R1181 unlock-stmt -> UNLOCK ( lock-variable [, sync-stat-list] )
TYPE_CONTEXT_PARSER("UNLOCK statement"_en_US,
    construct<UnlockStmt>("UNLOCK (" >> lockVariable,
        defaulted("," >> nonemptyList(statOrErrmsg)) / ")"))

// CUF-kernel-do-construct -> CUF-kernel-do-directive do-construct
// CUF-kernel-do-directive ->
//     !$CUF KERNEL DO [ (scalar-int-constant-expr) ] <<< grid, block [, stream]
//     >>> do-construct
// grid -> * | scalar-int-expr | ( scalar-int-expr-list )
// block -> * | scalar-int-expr | ( scalar-int-expr-list )
// stream -> ( 0, | STREAM = ) scalar-int-expr
TYPE_PARSER(sourced(beginDirective >> "$CUF KERNEL DO"_tok >>
    construct<CUFKernelDoConstruct::Directive>(
        maybe(parenthesized(scalarIntConstantExpr)),
        "<<<" >>
            ("*" >> pure<std::list<ScalarIntExpr>>() ||
                parenthesized(nonemptyList(scalarIntExpr)) ||
                applyFunction(singletonList<ScalarIntExpr>, scalarIntExpr)),
        "," >> ("*" >> pure<std::list<ScalarIntExpr>>() ||
                   parenthesized(nonemptyList(scalarIntExpr)) ||
                   applyFunction(singletonList<ScalarIntExpr>, scalarIntExpr)),
        maybe((", 0 ,"_tok || ", STREAM ="_tok) >> scalarIntExpr) / ">>>" /
            endDirective)))
TYPE_CONTEXT_PARSER("!$CUF KERNEL DO construct"_en_US,
    extension<LanguageFeature::CUDA>(construct<CUFKernelDoConstruct>(
        Parser<CUFKernelDoConstruct::Directive>{},
        maybe(Parser<DoConstruct>{}))))

} // namespace Fortran::parser
