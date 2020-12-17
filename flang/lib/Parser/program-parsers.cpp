//===-- lib/Parser/program-parsers.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Per-type parsers for program units

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

// R502 program-unit ->
//        main-program | external-subprogram | module | submodule | block-data
// R503 external-subprogram -> function-subprogram | subroutine-subprogram
// N.B. "module" must precede "external-subprogram" in this sequence of
// alternatives to avoid ambiguity with the MODULE keyword prefix that
// they recognize.  I.e., "modulesubroutinefoo" should start a module
// "subroutinefoo", not a subroutine "foo" with the MODULE prefix.  The
// ambiguity is exacerbated by the extension that accepts a function
// statement without an otherwise empty list of dummy arguments.  That
// MODULE prefix is disallowed by a constraint (C1547) in this context,
// so the standard language is not ambiguous, but disabling its misrecognition
// here would require context-sensitive keyword recognition or (or via)
// variant parsers for several productions; giving the "module" production
// priority here is a cleaner solution, though regrettably subtle.  Enforcing
// C1547 is done in semantics.
static constexpr auto programUnit{
    construct<ProgramUnit>(indirect(Parser<Module>{})) ||
    construct<ProgramUnit>(indirect(functionSubprogram)) ||
    construct<ProgramUnit>(indirect(subroutineSubprogram)) ||
    construct<ProgramUnit>(indirect(Parser<Submodule>{})) ||
    construct<ProgramUnit>(indirect(Parser<BlockData>{})) ||
    construct<ProgramUnit>(indirect(Parser<MainProgram>{}))};
static constexpr auto normalProgramUnit{StartNewSubprogram{} >> programUnit /
        skipMany(";"_tok) / space / recovery(endOfLine, SkipPast<'\n'>{})};
static constexpr auto globalCompilerDirective{
    construct<ProgramUnit>(indirect(compilerDirective))};

// R501 program -> program-unit [program-unit]...
// This is the top-level production for the Fortran language.
// F'2018 6.3.1 defines a program unit as a sequence of one or more lines,
// implying that a line can't be part of two distinct program units.
// Consequently, a program unit END statement should be the last statement
// on its line.  We parse those END statements via unterminatedStatement()
// and then skip over the end of the line here.
TYPE_PARSER(construct<Program>(
    extension<LanguageFeature::EmptySourceFile>(skipStuffBeforeStatement >>
        !nextCh >> pure<std::list<ProgramUnit>>()) ||
    some(globalCompilerDirective || normalProgramUnit) /
        skipStuffBeforeStatement))

// R504 specification-part ->
//         [use-stmt]... [import-stmt]... [implicit-part]
//         [declaration-construct]...
TYPE_CONTEXT_PARSER("specification part"_en_US,
    construct<SpecificationPart>(many(openaccDeclarativeConstruct),
        many(openmpDeclarativeConstruct), many(indirect(compilerDirective)),
        many(statement(indirect(Parser<UseStmt>{}))),
        many(unambiguousStatement(indirect(Parser<ImportStmt>{}))),
        implicitPart, many(declarationConstruct)))

// R507 declaration-construct ->
//        specification-construct | data-stmt | format-stmt |
//        entry-stmt | stmt-function-stmt
// N.B. These parsers incorporate recognition of some other statements that
// may have been misplaced in the sequence of statements that are acceptable
// as a specification part in order to improve error recovery.
// Also note that many instances of specification-part in the standard grammar
// are in contexts that impose constraints on the kinds of statements that
// are allowed, and so we have a variant production for declaration-construct
// that implements those constraints.
constexpr auto execPartLookAhead{
    first(actionStmt >> ok, openaccConstruct >> ok, openmpConstruct >> ok,
        "ASSOCIATE ("_tok, "BLOCK"_tok, "SELECT"_tok, "CHANGE TEAM"_sptok,
        "CRITICAL"_tok, "DO"_tok, "IF ("_tok, "WHERE ("_tok, "FORALL ("_tok)};
constexpr auto declErrorRecovery{
    stmtErrorRecoveryStart >> !execPartLookAhead >> skipStmtErrorRecovery};
constexpr auto misplacedSpecificationStmt{Parser<UseStmt>{} >>
        fail<DeclarationConstruct>("misplaced USE statement"_err_en_US) ||
    Parser<ImportStmt>{} >>
        fail<DeclarationConstruct>(
            "IMPORT statements must follow any USE statements and precede all other declarations"_err_en_US) ||
    Parser<ImplicitStmt>{} >>
        fail<DeclarationConstruct>(
            "IMPLICIT statements must follow USE and IMPORT and precede all other declarations"_err_en_US)};

TYPE_PARSER(recovery(
    withMessage("expected declaration construct"_err_en_US,
        CONTEXT_PARSER("declaration construct"_en_US,
            first(construct<DeclarationConstruct>(specificationConstruct),
                construct<DeclarationConstruct>(statement(indirect(dataStmt))),
                construct<DeclarationConstruct>(
                    statement(indirect(formatStmt))),
                construct<DeclarationConstruct>(statement(indirect(entryStmt))),
                construct<DeclarationConstruct>(
                    statement(indirect(Parser<StmtFunctionStmt>{}))),
                misplacedSpecificationStmt))),
    construct<DeclarationConstruct>(declErrorRecovery)))

// R507 variant of declaration-construct for use in limitedSpecificationPart.
constexpr auto invalidDeclarationStmt{formatStmt >>
        fail<DeclarationConstruct>(
            "FORMAT statements are not permitted in this specification part"_err_en_US) ||
    entryStmt >>
        fail<DeclarationConstruct>(
            "ENTRY statements are not permitted in this specification part"_err_en_US)};

constexpr auto limitedDeclarationConstruct{recovery(
    withMessage("expected declaration construct"_err_en_US,
        inContext("declaration construct"_en_US,
            first(construct<DeclarationConstruct>(specificationConstruct),
                construct<DeclarationConstruct>(statement(indirect(dataStmt))),
                misplacedSpecificationStmt, invalidDeclarationStmt))),
    construct<DeclarationConstruct>(
        stmtErrorRecoveryStart >> skipStmtErrorRecovery))};

// R504 variant for many contexts (modules, submodules, BLOCK DATA subprograms,
// and interfaces) which have constraints on their specification parts that
// preclude FORMAT, ENTRY, and statement functions, and benefit from
// specialized error recovery in the event of a spurious executable
// statement.
constexpr auto limitedSpecificationPart{inContext("specification part"_en_US,
    construct<SpecificationPart>(many(openaccDeclarativeConstruct),
        many(openmpDeclarativeConstruct), many(indirect(compilerDirective)),
        many(statement(indirect(Parser<UseStmt>{}))),
        many(unambiguousStatement(indirect(Parser<ImportStmt>{}))),
        implicitPart, many(limitedDeclarationConstruct)))};

// R508 specification-construct ->
//        derived-type-def | enum-def | generic-stmt | interface-block |
//        parameter-stmt | procedure-declaration-stmt |
//        other-specification-stmt | type-declaration-stmt
TYPE_CONTEXT_PARSER("specification construct"_en_US,
    first(construct<SpecificationConstruct>(indirect(Parser<DerivedTypeDef>{})),
        construct<SpecificationConstruct>(indirect(Parser<EnumDef>{})),
        construct<SpecificationConstruct>(
            statement(indirect(Parser<GenericStmt>{}))),
        construct<SpecificationConstruct>(indirect(interfaceBlock)),
        construct<SpecificationConstruct>(statement(indirect(parameterStmt))),
        construct<SpecificationConstruct>(
            statement(indirect(oldParameterStmt))),
        construct<SpecificationConstruct>(
            statement(indirect(Parser<ProcedureDeclarationStmt>{}))),
        construct<SpecificationConstruct>(
            statement(Parser<OtherSpecificationStmt>{})),
        construct<SpecificationConstruct>(
            statement(indirect(typeDeclarationStmt))),
        construct<SpecificationConstruct>(indirect(Parser<StructureDef>{})),
        construct<SpecificationConstruct>(
            indirect(openaccDeclarativeConstruct)),
        construct<SpecificationConstruct>(indirect(openmpDeclarativeConstruct)),
        construct<SpecificationConstruct>(indirect(compilerDirective))))

// R513 other-specification-stmt ->
//        access-stmt | allocatable-stmt | asynchronous-stmt | bind-stmt |
//        codimension-stmt | contiguous-stmt | dimension-stmt | external-stmt |
//        intent-stmt | intrinsic-stmt | namelist-stmt | optional-stmt |
//        pointer-stmt | protected-stmt | save-stmt | target-stmt |
//        volatile-stmt | value-stmt | common-stmt | equivalence-stmt
TYPE_PARSER(first(
    construct<OtherSpecificationStmt>(indirect(Parser<AccessStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<AllocatableStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<AsynchronousStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<BindStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<CodimensionStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<ContiguousStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<DimensionStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<ExternalStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<IntentStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<IntrinsicStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<NamelistStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<OptionalStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<PointerStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<ProtectedStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<SaveStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<TargetStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<ValueStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<VolatileStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<CommonStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<EquivalenceStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<BasedPointerStmt>{}))))

// R1401 main-program ->
//         [program-stmt] [specification-part] [execution-part]
//         [internal-subprogram-part] end-program-stmt
TYPE_CONTEXT_PARSER("main program"_en_US,
    construct<MainProgram>(maybe(statement(Parser<ProgramStmt>{})),
        specificationPart, executionPart, maybe(internalSubprogramPart),
        unterminatedStatement(Parser<EndProgramStmt>{})))

// R1402 program-stmt -> PROGRAM program-name
// PGI allows empty parentheses after the name.
TYPE_CONTEXT_PARSER("PROGRAM statement"_en_US,
    construct<ProgramStmt>("PROGRAM" >> name /
            maybe(extension<LanguageFeature::ProgramParentheses>(
                parenthesized(ok)))))

// R1403 end-program-stmt -> END [PROGRAM [program-name]]
TYPE_CONTEXT_PARSER("END PROGRAM statement"_en_US,
    construct<EndProgramStmt>(recovery(
        "END PROGRAM" >> maybe(name) || bareEnd, progUnitEndStmtErrorRecovery)))

// R1404 module ->
//         module-stmt [specification-part] [module-subprogram-part]
//         end-module-stmt
TYPE_CONTEXT_PARSER("module"_en_US,
    construct<Module>(statement(Parser<ModuleStmt>{}), limitedSpecificationPart,
        maybe(Parser<ModuleSubprogramPart>{}),
        unterminatedStatement(Parser<EndModuleStmt>{})))

// R1405 module-stmt -> MODULE module-name
TYPE_CONTEXT_PARSER(
    "MODULE statement"_en_US, construct<ModuleStmt>("MODULE" >> name))

// R1406 end-module-stmt -> END [MODULE [module-name]]
TYPE_CONTEXT_PARSER("END MODULE statement"_en_US,
    construct<EndModuleStmt>(recovery(
        "END MODULE" >> maybe(name) || bareEnd, progUnitEndStmtErrorRecovery)))

// R1407 module-subprogram-part -> contains-stmt [module-subprogram]...
TYPE_CONTEXT_PARSER("module subprogram part"_en_US,
    construct<ModuleSubprogramPart>(statement(containsStmt),
        many(StartNewSubprogram{} >> Parser<ModuleSubprogram>{})))

// R1408 module-subprogram ->
//         function-subprogram | subroutine-subprogram |
//         separate-module-subprogram
TYPE_PARSER(construct<ModuleSubprogram>(indirect(functionSubprogram)) ||
    construct<ModuleSubprogram>(indirect(subroutineSubprogram)) ||
    construct<ModuleSubprogram>(indirect(Parser<SeparateModuleSubprogram>{})))

// R1410 module-nature -> INTRINSIC | NON_INTRINSIC
constexpr auto moduleNature{
    "INTRINSIC" >> pure(UseStmt::ModuleNature::Intrinsic) ||
    "NON_INTRINSIC" >> pure(UseStmt::ModuleNature::Non_Intrinsic)};

// R1409 use-stmt ->
//         USE [[, module-nature] ::] module-name [, rename-list] |
//         USE [[, module-nature] ::] module-name , ONLY : [only-list]
// N.B. Lookahead to the end of the statement is necessary to resolve
// ambiguity with assignments and statement function definitions that
// begin with the letters "USE".
TYPE_PARSER(construct<UseStmt>("USE" >> optionalBeforeColons(moduleNature),
                name, ", ONLY :" >> optionalList(Parser<Only>{})) ||
    construct<UseStmt>("USE" >> optionalBeforeColons(moduleNature), name,
        defaulted("," >>
            nonemptyList("expected renamings"_err_en_US, Parser<Rename>{})) /
            lookAhead(endOfStmt)))

// R1411 rename ->
//         local-name => use-name |
//         OPERATOR ( local-defined-operator ) =>
//           OPERATOR ( use-defined-operator )
TYPE_PARSER(construct<Rename>("OPERATOR (" >>
                construct<Rename::Operators>(
                    definedOpName / ") => OPERATOR (", definedOpName / ")")) ||
    construct<Rename>(construct<Rename::Names>(name, "=>" >> name)))

// R1412 only -> generic-spec | only-use-name | rename
// R1413 only-use-name -> use-name
// N.B. generic-spec and only-use-name are ambiguous; resolved with symbols
TYPE_PARSER(construct<Only>(Parser<Rename>{}) ||
    construct<Only>(indirect(genericSpec)) || construct<Only>(name))

// R1416 submodule ->
//         submodule-stmt [specification-part] [module-subprogram-part]
//         end-submodule-stmt
TYPE_CONTEXT_PARSER("submodule"_en_US,
    construct<Submodule>(statement(Parser<SubmoduleStmt>{}),
        limitedSpecificationPart, maybe(Parser<ModuleSubprogramPart>{}),
        unterminatedStatement(Parser<EndSubmoduleStmt>{})))

// R1417 submodule-stmt -> SUBMODULE ( parent-identifier ) submodule-name
TYPE_CONTEXT_PARSER("SUBMODULE statement"_en_US,
    construct<SubmoduleStmt>(
        "SUBMODULE" >> parenthesized(Parser<ParentIdentifier>{}), name))

// R1418 parent-identifier -> ancestor-module-name [: parent-submodule-name]
TYPE_PARSER(construct<ParentIdentifier>(name, maybe(":" >> name)))

// R1419 end-submodule-stmt -> END [SUBMODULE [submodule-name]]
TYPE_CONTEXT_PARSER("END SUBMODULE statement"_en_US,
    construct<EndSubmoduleStmt>(
        recovery("END SUBMODULE" >> maybe(name) || bareEnd,
            progUnitEndStmtErrorRecovery)))

// R1420 block-data -> block-data-stmt [specification-part] end-block-data-stmt
TYPE_CONTEXT_PARSER("BLOCK DATA subprogram"_en_US,
    construct<BlockData>(statement(Parser<BlockDataStmt>{}),
        limitedSpecificationPart,
        unterminatedStatement(Parser<EndBlockDataStmt>{})))

// R1421 block-data-stmt -> BLOCK DATA [block-data-name]
TYPE_CONTEXT_PARSER("BLOCK DATA statement"_en_US,
    construct<BlockDataStmt>("BLOCK DATA" >> maybe(name)))

// R1422 end-block-data-stmt -> END [BLOCK DATA [block-data-name]]
TYPE_CONTEXT_PARSER("END BLOCK DATA statement"_en_US,
    construct<EndBlockDataStmt>(
        recovery("END BLOCK DATA" >> maybe(name) || bareEnd,
            progUnitEndStmtErrorRecovery)))

// R1501 interface-block ->
//         interface-stmt [interface-specification]... end-interface-stmt
TYPE_PARSER(construct<InterfaceBlock>(statement(Parser<InterfaceStmt>{}),
    many(Parser<InterfaceSpecification>{}),
    statement(Parser<EndInterfaceStmt>{})))

// R1502 interface-specification -> interface-body | procedure-stmt
TYPE_PARSER(construct<InterfaceSpecification>(Parser<InterfaceBody>{}) ||
    construct<InterfaceSpecification>(statement(Parser<ProcedureStmt>{})))

// R1503 interface-stmt -> INTERFACE [generic-spec] | ABSTRACT INTERFACE
TYPE_PARSER(construct<InterfaceStmt>("INTERFACE" >> maybe(genericSpec)) ||
    construct<InterfaceStmt>(construct<Abstract>("ABSTRACT INTERFACE"_sptok)))

// R1504 end-interface-stmt -> END INTERFACE [generic-spec]
TYPE_PARSER(construct<EndInterfaceStmt>("END INTERFACE" >> maybe(genericSpec)))

// R1505 interface-body ->
//         function-stmt [specification-part] end-function-stmt |
//         subroutine-stmt [specification-part] end-subroutine-stmt
TYPE_CONTEXT_PARSER("interface body"_en_US,
    construct<InterfaceBody>(
        construct<InterfaceBody::Function>(statement(functionStmt),
            indirect(limitedSpecificationPart), statement(endFunctionStmt))) ||
        construct<InterfaceBody>(construct<InterfaceBody::Subroutine>(
            statement(subroutineStmt), indirect(limitedSpecificationPart),
            statement(endSubroutineStmt))))

// R1507 specific-procedure -> procedure-name
constexpr auto specificProcedures{
    nonemptyList("expected specific procedure names"_err_en_US, name)};

// R1506 procedure-stmt -> [MODULE] PROCEDURE [::] specific-procedure-list
TYPE_PARSER(construct<ProcedureStmt>("MODULE PROCEDURE"_sptok >>
                    pure(ProcedureStmt::Kind::ModuleProcedure),
                maybe("::"_tok) >> specificProcedures) ||
    construct<ProcedureStmt>(
        "PROCEDURE" >> pure(ProcedureStmt::Kind::Procedure),
        maybe("::"_tok) >> specificProcedures))

// R1508 generic-spec ->
//         generic-name | OPERATOR ( defined-operator ) |
//         ASSIGNMENT ( = ) | defined-io-generic-spec
// R1509 defined-io-generic-spec ->
//         READ ( FORMATTED ) | READ ( UNFORMATTED ) |
//         WRITE ( FORMATTED ) | WRITE ( UNFORMATTED )
TYPE_PARSER(sourced(first(construct<GenericSpec>("OPERATOR" >>
                              parenthesized(Parser<DefinedOperator>{})),
    construct<GenericSpec>(
        construct<GenericSpec::Assignment>("ASSIGNMENT ( = )"_tok)),
    construct<GenericSpec>(
        construct<GenericSpec::ReadFormatted>("READ ( FORMATTED )"_tok)),
    construct<GenericSpec>(
        construct<GenericSpec::ReadUnformatted>("READ ( UNFORMATTED )"_tok)),
    construct<GenericSpec>(
        construct<GenericSpec::WriteFormatted>("WRITE ( FORMATTED )"_tok)),
    construct<GenericSpec>(
        construct<GenericSpec::WriteUnformatted>("WRITE ( UNFORMATTED )"_tok)),
    construct<GenericSpec>(name))))

// R1510 generic-stmt ->
//         GENERIC [, access-spec] :: generic-spec => specific-procedure-list
TYPE_PARSER(construct<GenericStmt>("GENERIC" >> maybe("," >> accessSpec),
    "::" >> genericSpec, "=>" >> specificProcedures))

// R1511 external-stmt -> EXTERNAL [::] external-name-list
TYPE_PARSER(
    "EXTERNAL" >> maybe("::"_tok) >> construct<ExternalStmt>(listOfNames))

// R1512 procedure-declaration-stmt ->
//         PROCEDURE ( [proc-interface] ) [[, proc-attr-spec]... ::]
//         proc-decl-list
TYPE_PARSER("PROCEDURE" >>
    construct<ProcedureDeclarationStmt>(parenthesized(maybe(procInterface)),
        optionalListBeforeColons(Parser<ProcAttrSpec>{}),
        nonemptyList("expected procedure declarations"_err_en_US, procDecl)))

// R1513 proc-interface -> interface-name | declaration-type-spec
// R1516 interface-name -> name
// N.B. Simple names of intrinsic types (e.g., "REAL") are not
// ambiguous here - they take precedence over derived type names
// thanks to C1516.
TYPE_PARSER(
    construct<ProcInterface>(declarationTypeSpec / lookAhead(")"_tok)) ||
    construct<ProcInterface>(name))

// R1514 proc-attr-spec ->
//         access-spec | proc-language-binding-spec | INTENT ( intent-spec ) |
//         OPTIONAL | POINTER | PROTECTED | SAVE
TYPE_PARSER(construct<ProcAttrSpec>(accessSpec) ||
    construct<ProcAttrSpec>(languageBindingSpec) ||
    construct<ProcAttrSpec>("INTENT" >> parenthesized(intentSpec)) ||
    construct<ProcAttrSpec>(optional) || construct<ProcAttrSpec>(pointer) ||
    construct<ProcAttrSpec>(protectedAttr) || construct<ProcAttrSpec>(save))

// R1515 proc-decl -> procedure-entity-name [=> proc-pointer-init]
TYPE_PARSER(construct<ProcDecl>(name, maybe("=>" >> Parser<ProcPointerInit>{})))

// R1517 proc-pointer-init -> null-init | initial-proc-target
// R1518 initial-proc-target -> procedure-name
TYPE_PARSER(
    construct<ProcPointerInit>(nullInit) || construct<ProcPointerInit>(name))

// R1519 intrinsic-stmt -> INTRINSIC [::] intrinsic-procedure-name-list
TYPE_PARSER(
    "INTRINSIC" >> maybe("::"_tok) >> construct<IntrinsicStmt>(listOfNames))

// R1520 function-reference -> procedure-designator ( [actual-arg-spec-list] )
TYPE_CONTEXT_PARSER("function reference"_en_US,
    construct<FunctionReference>(
        sourced(construct<Call>(Parser<ProcedureDesignator>{},
            parenthesized(optionalList(actualArgSpec))))) /
        !"["_tok)

// R1521 call-stmt -> CALL procedure-designator [( [actual-arg-spec-list] )]
TYPE_PARSER(construct<CallStmt>(
    sourced(construct<Call>("CALL" >> Parser<ProcedureDesignator>{},
        defaulted(parenthesized(optionalList(actualArgSpec)))))))

// R1522 procedure-designator ->
//         procedure-name | proc-component-ref | data-ref % binding-name
TYPE_PARSER(construct<ProcedureDesignator>(Parser<ProcComponentRef>{}) ||
    construct<ProcedureDesignator>(name))

// R1523 actual-arg-spec -> [keyword =] actual-arg
TYPE_PARSER(construct<ActualArgSpec>(
    maybe(keyword / "=" / !"="_ch), Parser<ActualArg>{}))

// R1524 actual-arg ->
//         expr | variable | procedure-name | proc-component-ref |
//         alt-return-spec
// N.B. the "procedure-name" and "proc-component-ref" alternatives can't
// yet be distinguished from "variable", many instances of which can't be
// distinguished from "expr" anyway (to do so would misparse structure
// constructors and function calls as array elements).
// Semantics sorts it all out later.
TYPE_PARSER(construct<ActualArg>(expr) ||
    construct<ActualArg>(Parser<AltReturnSpec>{}) ||
    extension<LanguageFeature::PercentRefAndVal>(construct<ActualArg>(
        construct<ActualArg::PercentRef>("%REF" >> parenthesized(variable)))) ||
    extension<LanguageFeature::PercentRefAndVal>(construct<ActualArg>(
        construct<ActualArg::PercentVal>("%VAL" >> parenthesized(expr)))))

// R1525 alt-return-spec -> * label
TYPE_PARSER(construct<AltReturnSpec>(star >> label))

// R1527 prefix-spec ->
//         declaration-type-spec | ELEMENTAL | IMPURE | MODULE |
//         NON_RECURSIVE | PURE | RECURSIVE
TYPE_PARSER(first(construct<PrefixSpec>(declarationTypeSpec),
    construct<PrefixSpec>(construct<PrefixSpec::Elemental>("ELEMENTAL"_tok)),
    construct<PrefixSpec>(construct<PrefixSpec::Impure>("IMPURE"_tok)),
    construct<PrefixSpec>(construct<PrefixSpec::Module>("MODULE"_tok)),
    construct<PrefixSpec>(
        construct<PrefixSpec::Non_Recursive>("NON_RECURSIVE"_tok)),
    construct<PrefixSpec>(construct<PrefixSpec::Pure>("PURE"_tok)),
    construct<PrefixSpec>(construct<PrefixSpec::Recursive>("RECURSIVE"_tok))))

// R1529 function-subprogram ->
//         function-stmt [specification-part] [execution-part]
//         [internal-subprogram-part] end-function-stmt
TYPE_CONTEXT_PARSER("FUNCTION subprogram"_en_US,
    construct<FunctionSubprogram>(statement(functionStmt), specificationPart,
        executionPart, maybe(internalSubprogramPart),
        unterminatedStatement(endFunctionStmt)))

// R1530 function-stmt ->
//         [prefix] FUNCTION function-name ( [dummy-arg-name-list] ) [suffix]
// R1526 prefix -> prefix-spec [prefix-spec]...
// R1531 dummy-arg-name -> name
TYPE_CONTEXT_PARSER("FUNCTION statement"_en_US,
    construct<FunctionStmt>(many(prefixSpec), "FUNCTION" >> name,
        parenthesized(optionalList(name)), maybe(suffix)) ||
        extension<LanguageFeature::OmitFunctionDummies>(
            construct<FunctionStmt>( // PGI & Intel accept "FUNCTION F"
                many(prefixSpec), "FUNCTION" >> name,
                construct<std::list<Name>>(),
                construct<std::optional<Suffix>>())))

// R1532 suffix ->
//         proc-language-binding-spec [RESULT ( result-name )] |
//         RESULT ( result-name ) [proc-language-binding-spec]
TYPE_PARSER(construct<Suffix>(
                languageBindingSpec, maybe("RESULT" >> parenthesized(name))) ||
    construct<Suffix>(
        "RESULT" >> parenthesized(name), maybe(languageBindingSpec)))

// R1533 end-function-stmt -> END [FUNCTION [function-name]]
TYPE_PARSER(construct<EndFunctionStmt>(recovery(
    "END FUNCTION" >> maybe(name) || bareEnd, progUnitEndStmtErrorRecovery)))

// R1534 subroutine-subprogram ->
//         subroutine-stmt [specification-part] [execution-part]
//         [internal-subprogram-part] end-subroutine-stmt
TYPE_CONTEXT_PARSER("SUBROUTINE subprogram"_en_US,
    construct<SubroutineSubprogram>(statement(subroutineStmt),
        specificationPart, executionPart, maybe(internalSubprogramPart),
        unterminatedStatement(endSubroutineStmt)))

// R1535 subroutine-stmt ->
//         [prefix] SUBROUTINE subroutine-name [( [dummy-arg-list] )
//         [proc-language-binding-spec]]
TYPE_PARSER(
    construct<SubroutineStmt>(many(prefixSpec), "SUBROUTINE" >> name,
        parenthesized(optionalList(dummyArg)), maybe(languageBindingSpec)) ||
    construct<SubroutineStmt>(many(prefixSpec), "SUBROUTINE" >> name,
        pure<std::list<DummyArg>>(),
        pure<std::optional<LanguageBindingSpec>>()))

// R1536 dummy-arg -> dummy-arg-name | *
TYPE_PARSER(construct<DummyArg>(name) || construct<DummyArg>(star))

// R1537 end-subroutine-stmt -> END [SUBROUTINE [subroutine-name]]
TYPE_PARSER(construct<EndSubroutineStmt>(recovery(
    "END SUBROUTINE" >> maybe(name) || bareEnd, progUnitEndStmtErrorRecovery)))

// R1538 separate-module-subprogram ->
//         mp-subprogram-stmt [specification-part] [execution-part]
//         [internal-subprogram-part] end-mp-subprogram-stmt
TYPE_CONTEXT_PARSER("separate module subprogram"_en_US,
    construct<SeparateModuleSubprogram>(statement(Parser<MpSubprogramStmt>{}),
        specificationPart, executionPart, maybe(internalSubprogramPart),
        statement(Parser<EndMpSubprogramStmt>{})))

// R1539 mp-subprogram-stmt -> MODULE PROCEDURE procedure-name
TYPE_CONTEXT_PARSER("MODULE PROCEDURE statement"_en_US,
    construct<MpSubprogramStmt>("MODULE PROCEDURE"_sptok >> name))

// R1540 end-mp-subprogram-stmt -> END [PROCEDURE [procedure-name]]
TYPE_CONTEXT_PARSER("END PROCEDURE statement"_en_US,
    construct<EndMpSubprogramStmt>(
        recovery("END PROCEDURE" >> maybe(name) || bareEnd,
            progUnitEndStmtErrorRecovery)))

// R1541 entry-stmt -> ENTRY entry-name [( [dummy-arg-list] ) [suffix]]
TYPE_PARSER(
    "ENTRY" >> (construct<EntryStmt>(name,
                    parenthesized(optionalList(dummyArg)), maybe(suffix)) ||
                   construct<EntryStmt>(name, construct<std::list<DummyArg>>(),
                       construct<std::optional<Suffix>>())))

// R1542 return-stmt -> RETURN [scalar-int-expr]
TYPE_CONTEXT_PARSER("RETURN statement"_en_US,
    construct<ReturnStmt>("RETURN" >> maybe(scalarIntExpr)))

// R1543 contains-stmt -> CONTAINS
TYPE_PARSER(construct<ContainsStmt>("CONTAINS"_tok))

// R1544 stmt-function-stmt ->
//         function-name ( [dummy-arg-name-list] ) = scalar-expr
TYPE_CONTEXT_PARSER("statement function definition"_en_US,
    construct<StmtFunctionStmt>(
        name, parenthesized(optionalList(name)), "=" >> scalar(expr)))
} // namespace Fortran::parser
