//===-- lib/Parser/Fortran-parsers.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Top-level grammar specification for Fortran.  These parsers drive
// the tokenization parsers in cooked-tokens.h to consume characters,
// recognize the productions of Fortran, and to construct a parse tree.
// See ParserCombinators.md for documentation on the parser combinator
// library used here to implement an LL recursive descent recognizer.

// The productions that follow are derived from the draft Fortran 2018
// standard, with some necessary modifications to remove left recursion
// and some generalization in order to defer cases where parses depend
// on the definitions of symbols.  The "Rxxx" numbers that appear in
// comments refer to these numbered requirements in the Fortran standard.

// The whole Fortran grammar originally constituted one header file,
// but that turned out to require more memory to compile with current
// C++ compilers than some people were willing to accept, so now the
// various per-type parsers are partitioned into several C++ source
// files.  This file contains parsers for constants, types, declarations,
// and misfits (mostly clauses 7, 8, & 9 of Fortran 2018).  The others:
//  executable-parsers.cpp  Executable statements
//  expr-parsers.cpp        Expressions
//  io-parsers.cpp          I/O statements and FORMAT
//  openmp-parsers.cpp      OpenMP directives
//  program-parsers.cpp     Program units

#include "basic-parsers.h"
#include "expr-parsers.h"
#include "misc-parsers.h"
#include "stmt-parser.h"
#include "token-parsers.h"
#include "type-parser-implementation.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Parser/user-state.h"

namespace Fortran::parser {

// R601 alphanumeric-character -> letter | digit | underscore
// R603 name -> letter [alphanumeric-character]...
constexpr auto nonDigitIdChar{letter || otherIdChar};
constexpr auto rawName{nonDigitIdChar >> many(nonDigitIdChar || digit)};
TYPE_PARSER(space >> sourced(rawName >> construct<Name>()))

// R608 intrinsic-operator ->
//        power-op | mult-op | add-op | concat-op | rel-op |
//        not-op | and-op | or-op | equiv-op
// R610 extended-intrinsic-op -> intrinsic-operator
// These parsers must be ordered carefully to avoid misrecognition.
constexpr auto namedIntrinsicOperator{
    ".LT." >> pure(DefinedOperator::IntrinsicOperator::LT) ||
    ".LE." >> pure(DefinedOperator::IntrinsicOperator::LE) ||
    ".EQ." >> pure(DefinedOperator::IntrinsicOperator::EQ) ||
    ".NE." >> pure(DefinedOperator::IntrinsicOperator::NE) ||
    ".GE." >> pure(DefinedOperator::IntrinsicOperator::GE) ||
    ".GT." >> pure(DefinedOperator::IntrinsicOperator::GT) ||
    ".NOT." >> pure(DefinedOperator::IntrinsicOperator::NOT) ||
    ".AND." >> pure(DefinedOperator::IntrinsicOperator::AND) ||
    ".OR." >> pure(DefinedOperator::IntrinsicOperator::OR) ||
    ".EQV." >> pure(DefinedOperator::IntrinsicOperator::EQV) ||
    ".NEQV." >> pure(DefinedOperator::IntrinsicOperator::NEQV) ||
    extension<LanguageFeature::XOROperator>(
        "nonstandard usage: .XOR. spelling of .NEQV."_port_en_US,
        ".XOR." >> pure(DefinedOperator::IntrinsicOperator::NEQV)) ||
    extension<LanguageFeature::LogicalAbbreviations>(
        "nonstandard usage: abbreviated logical operator"_port_en_US,
        ".N." >> pure(DefinedOperator::IntrinsicOperator::NOT) ||
            ".A." >> pure(DefinedOperator::IntrinsicOperator::AND) ||
            ".O." >> pure(DefinedOperator::IntrinsicOperator::OR) ||
            extension<LanguageFeature::XOROperator>(
                "nonstandard usage: .X. spelling of .NEQV."_port_en_US,
                ".X." >> pure(DefinedOperator::IntrinsicOperator::NEQV)))};

constexpr auto intrinsicOperator{
    "**" >> pure(DefinedOperator::IntrinsicOperator::Power) ||
    "*" >> pure(DefinedOperator::IntrinsicOperator::Multiply) ||
    "//" >> pure(DefinedOperator::IntrinsicOperator::Concat) ||
    "/=" >> pure(DefinedOperator::IntrinsicOperator::NE) ||
    "/" >> pure(DefinedOperator::IntrinsicOperator::Divide) ||
    "+" >> pure(DefinedOperator::IntrinsicOperator::Add) ||
    "-" >> pure(DefinedOperator::IntrinsicOperator::Subtract) ||
    "<=" >> pure(DefinedOperator::IntrinsicOperator::LE) ||
    extension<LanguageFeature::AlternativeNE>(
        "nonstandard usage: <> spelling of /= or .NE."_port_en_US,
        "<>" >> pure(DefinedOperator::IntrinsicOperator::NE)) ||
    "<" >> pure(DefinedOperator::IntrinsicOperator::LT) ||
    "==" >> pure(DefinedOperator::IntrinsicOperator::EQ) ||
    ">=" >> pure(DefinedOperator::IntrinsicOperator::GE) ||
    ">" >> pure(DefinedOperator::IntrinsicOperator::GT) ||
    namedIntrinsicOperator};

// R609 defined-operator ->
//        defined-unary-op | defined-binary-op | extended-intrinsic-op
TYPE_PARSER(construct<DefinedOperator>(intrinsicOperator) ||
    construct<DefinedOperator>(definedOpName))

// R505 implicit-part -> [implicit-part-stmt]... implicit-stmt
// N.B. PARAMETER, FORMAT, & ENTRY statements that appear before any
// other kind of declaration-construct will be parsed into the
// implicit-part.
TYPE_CONTEXT_PARSER("implicit part"_en_US,
    construct<ImplicitPart>(many(Parser<ImplicitPartStmt>{})))

// R506 implicit-part-stmt ->
//         implicit-stmt | parameter-stmt | format-stmt | entry-stmt
TYPE_PARSER(first(
    construct<ImplicitPartStmt>(statement(indirect(Parser<ImplicitStmt>{}))),
    construct<ImplicitPartStmt>(statement(indirect(parameterStmt))),
    construct<ImplicitPartStmt>(statement(indirect(oldParameterStmt))),
    construct<ImplicitPartStmt>(statement(indirect(formatStmt))),
    construct<ImplicitPartStmt>(statement(indirect(entryStmt))),
    construct<ImplicitPartStmt>(indirect(compilerDirective)),
    construct<ImplicitPartStmt>(indirect(openaccDeclarativeConstruct))))

// R512 internal-subprogram -> function-subprogram | subroutine-subprogram
// Internal subprograms are not program units, so their END statements
// can be followed by ';' and another statement on the same line.
TYPE_CONTEXT_PARSER("internal subprogram"_en_US,
    (construct<InternalSubprogram>(indirect(functionSubprogram)) ||
        construct<InternalSubprogram>(indirect(subroutineSubprogram))) /
            forceEndOfStmt ||
        construct<InternalSubprogram>(indirect(compilerDirective)))

// R511 internal-subprogram-part -> contains-stmt [internal-subprogram]...
TYPE_CONTEXT_PARSER("internal subprogram part"_en_US,
    construct<InternalSubprogramPart>(statement(containsStmt),
        many(StartNewSubprogram{} >> Parser<InternalSubprogram>{})))

// R605 literal-constant ->
//        int-literal-constant | real-literal-constant |
//        complex-literal-constant | logical-literal-constant |
//        char-literal-constant | boz-literal-constant
TYPE_PARSER(
    first(construct<LiteralConstant>(Parser<HollerithLiteralConstant>{}),
        construct<LiteralConstant>(realLiteralConstant),
        construct<LiteralConstant>(intLiteralConstant),
        construct<LiteralConstant>(Parser<ComplexLiteralConstant>{}),
        construct<LiteralConstant>(Parser<BOZLiteralConstant>{}),
        construct<LiteralConstant>(charLiteralConstant),
        construct<LiteralConstant>(Parser<LogicalLiteralConstant>{})))

// R606 named-constant -> name
TYPE_PARSER(construct<NamedConstant>(name))

// R701 type-param-value -> scalar-int-expr | * | :
TYPE_PARSER(construct<TypeParamValue>(scalarIntExpr) ||
    construct<TypeParamValue>(star) ||
    construct<TypeParamValue>(construct<TypeParamValue::Deferred>(":"_tok)))

// R702 type-spec -> intrinsic-type-spec | derived-type-spec
// N.B. This type-spec production is one of two instances in the Fortran
// grammar where intrinsic types and bare derived type names can clash;
// the other is below in R703 declaration-type-spec.  Look-ahead is required
// to disambiguate the cases where a derived type name begins with the name
// of an intrinsic type, e.g., REALITY.
TYPE_CONTEXT_PARSER("type spec"_en_US,
    construct<TypeSpec>(intrinsicTypeSpec / lookAhead("::"_tok || ")"_tok)) ||
        construct<TypeSpec>(derivedTypeSpec))

// R703 declaration-type-spec ->
//        intrinsic-type-spec | TYPE ( intrinsic-type-spec ) |
//        TYPE ( derived-type-spec ) | CLASS ( derived-type-spec ) |
//        CLASS ( * ) | TYPE ( * )
// N.B. It is critical to distribute "parenthesized()" over the alternatives
// for TYPE (...), rather than putting the alternatives within it, which
// would fail on "TYPE(real_derived)" with a misrecognition of "real" as an
// intrinsic-type-spec.
// N.B. TYPE(x) is a derived type if x is a one-word extension intrinsic
// type (BYTE or DOUBLECOMPLEX), not the extension intrinsic type.
TYPE_CONTEXT_PARSER("declaration type spec"_en_US,
    construct<DeclarationTypeSpec>(intrinsicTypeSpec) ||
        "TYPE" >>
            (parenthesized(construct<DeclarationTypeSpec>(
                 !"DOUBLECOMPLEX"_tok >> !"BYTE"_tok >> intrinsicTypeSpec)) ||
                parenthesized(construct<DeclarationTypeSpec>(
                    construct<DeclarationTypeSpec::Type>(derivedTypeSpec))) ||
                construct<DeclarationTypeSpec>(
                    "( * )" >> construct<DeclarationTypeSpec::TypeStar>())) ||
        "CLASS" >> parenthesized(construct<DeclarationTypeSpec>(
                                     construct<DeclarationTypeSpec::Class>(
                                         derivedTypeSpec)) ||
                       construct<DeclarationTypeSpec>("*" >>
                           construct<DeclarationTypeSpec::ClassStar>())) ||
        extension<LanguageFeature::DECStructures>(
            "nonstandard usage: STRUCTURE"_port_en_US,
            construct<DeclarationTypeSpec>(
                // As is also done for the STRUCTURE statement, the name of
                // the structure includes the surrounding slashes to avoid
                // name clashes.
                construct<DeclarationTypeSpec::Record>(
                    "RECORD" >> sourced("/" >> name / "/")))) ||
        construct<DeclarationTypeSpec>(vectorTypeSpec))

// R704 intrinsic-type-spec ->
//        integer-type-spec | REAL [kind-selector] | DOUBLE PRECISION |
//        COMPLEX [kind-selector] | CHARACTER [char-selector] |
//        LOGICAL [kind-selector]
// Extensions: DOUBLE COMPLEX, BYTE
TYPE_CONTEXT_PARSER("intrinsic type spec"_en_US,
    first(construct<IntrinsicTypeSpec>(integerTypeSpec),
        construct<IntrinsicTypeSpec>(
            construct<IntrinsicTypeSpec::Real>("REAL" >> maybe(kindSelector))),
        construct<IntrinsicTypeSpec>("DOUBLE PRECISION" >>
            construct<IntrinsicTypeSpec::DoublePrecision>()),
        construct<IntrinsicTypeSpec>(construct<IntrinsicTypeSpec::Complex>(
            "COMPLEX" >> maybe(kindSelector))),
        construct<IntrinsicTypeSpec>(construct<IntrinsicTypeSpec::Character>(
            "CHARACTER" >> maybe(Parser<CharSelector>{}))),
        construct<IntrinsicTypeSpec>(construct<IntrinsicTypeSpec::Logical>(
            "LOGICAL" >> maybe(kindSelector))),
        extension<LanguageFeature::DoubleComplex>(
            "nonstandard usage: DOUBLE COMPLEX"_port_en_US,
            construct<IntrinsicTypeSpec>("DOUBLE COMPLEX"_sptok >>
                construct<IntrinsicTypeSpec::DoubleComplex>())),
        extension<LanguageFeature::Byte>("nonstandard usage: BYTE"_port_en_US,
            construct<IntrinsicTypeSpec>(construct<IntegerTypeSpec>(
                "BYTE" >> construct<std::optional<KindSelector>>(pure(1)))))))

// Extension: Vector type
// VECTOR(intrinsic-type-spec) | __VECTOR_PAIR | __VECTOR_QUAD
TYPE_CONTEXT_PARSER("vector type spec"_en_US,
    extension<LanguageFeature::PPCVector>(
        "nonstandard usage: Vector type"_port_en_US,
        first(construct<VectorTypeSpec>(intrinsicVectorTypeSpec),
            construct<VectorTypeSpec>("__VECTOR_PAIR" >>
                construct<VectorTypeSpec::PairVectorTypeSpec>()),
            construct<VectorTypeSpec>("__VECTOR_QUAD" >>
                construct<VectorTypeSpec::QuadVectorTypeSpec>()))))

// VECTOR(integer-type-spec) | VECTOR(real-type-spec) |
// VECTOR(unsigend-type-spec) |
TYPE_PARSER(construct<IntrinsicVectorTypeSpec>("VECTOR" >>
    parenthesized(construct<VectorElementType>(integerTypeSpec) ||
        construct<VectorElementType>(unsignedTypeSpec) ||
        construct<VectorElementType>(construct<IntrinsicTypeSpec::Real>(
            "REAL" >> maybe(kindSelector))))))

// UNSIGNED type
TYPE_PARSER(construct<UnsignedTypeSpec>("UNSIGNED" >> maybe(kindSelector)))

// R705 integer-type-spec -> INTEGER [kind-selector]
TYPE_PARSER(construct<IntegerTypeSpec>("INTEGER" >> maybe(kindSelector)))

// R706 kind-selector -> ( [KIND =] scalar-int-constant-expr )
// Legacy extension: kind-selector -> * digit-string
TYPE_PARSER(construct<KindSelector>(
                parenthesized(maybe("KIND ="_tok) >> scalarIntConstantExpr)) ||
    extension<LanguageFeature::StarKind>(
        "nonstandard usage: TYPE*KIND syntax"_port_en_US,
        construct<KindSelector>(construct<KindSelector::StarSize>(
            "*" >> digitString64 / spaceCheck))))

constexpr auto noSpace{
    recovery(withMessage("invalid space"_err_en_US, !" "_ch), space)};

// R707 signed-int-literal-constant -> [sign] int-literal-constant
TYPE_PARSER(sourced(
    construct<SignedIntLiteralConstant>(SignedIntLiteralConstantWithoutKind{},
        maybe(noSpace >> underscore >> noSpace >> kindParam))))

// R708 int-literal-constant -> digit-string [_ kind-param]
// The negated look-ahead for a trailing underscore prevents misrecognition
// when the digit string is a numeric kind parameter of a character literal.
TYPE_PARSER(construct<IntLiteralConstant>(space >> digitString,
    maybe(underscore >> noSpace >> kindParam) / !underscore))

// R709 kind-param -> digit-string | scalar-int-constant-name
TYPE_PARSER(construct<KindParam>(digitString64) ||
    construct<KindParam>(
        scalar(integer(constant(sourced(rawName >> construct<Name>()))))))

// R712 sign -> + | -
// N.B. A sign constitutes a whole token, so a space is allowed in free form
// after the sign and before a real-literal-constant or
// complex-literal-constant.  A sign is not a unary operator in these contexts.
constexpr auto sign{
    "+"_tok >> pure(Sign::Positive) || "-"_tok >> pure(Sign::Negative)};

// R713 signed-real-literal-constant -> [sign] real-literal-constant
constexpr auto signedRealLiteralConstant{
    construct<SignedRealLiteralConstant>(maybe(sign), realLiteralConstant)};

// R714 real-literal-constant ->
//        significand [exponent-letter exponent] [_ kind-param] |
//        digit-string exponent-letter exponent [_ kind-param]
// R715 significand -> digit-string . [digit-string] | . digit-string
// R716 exponent-letter -> E | D
// Extension: Q
// R717 exponent -> signed-digit-string
constexpr auto exponentPart{
    ("ed"_ch ||
        extension<LanguageFeature::QuadPrecision>(
            "nonstandard usage: Q exponent"_port_en_US, "q"_ch)) >>
    SignedDigitString{}};

TYPE_CONTEXT_PARSER("REAL literal constant"_en_US,
    space >>
        construct<RealLiteralConstant>(
            sourced((digitString >> "."_ch >>
                            !(some(letter) >>
                                "."_ch /* don't misinterpret 1.AND. */) >>
                            maybe(digitString) >> maybe(exponentPart) >> ok ||
                        "."_ch >> digitString >> maybe(exponentPart) >> ok ||
                        digitString >> exponentPart >> ok) >>
                construct<RealLiteralConstant::Real>()),
            maybe(noSpace >> underscore >> noSpace >> kindParam)))

// R718 complex-literal-constant -> ( real-part , imag-part )
TYPE_CONTEXT_PARSER("COMPLEX literal constant"_en_US,
    parenthesized(construct<ComplexLiteralConstant>(
        Parser<ComplexPart>{} / ",", Parser<ComplexPart>{})))

// PGI/Intel extension: signed complex literal constant
TYPE_PARSER(construct<SignedComplexLiteralConstant>(
    sign, Parser<ComplexLiteralConstant>{}))

// R719 real-part ->
//        signed-int-literal-constant | signed-real-literal-constant |
//        named-constant
// R720 imag-part ->
//        signed-int-literal-constant | signed-real-literal-constant |
//        named-constant
TYPE_PARSER(construct<ComplexPart>(signedRealLiteralConstant) ||
    construct<ComplexPart>(signedIntLiteralConstant) ||
    construct<ComplexPart>(namedConstant))

// R721 char-selector ->
//        length-selector |
//        ( LEN = type-param-value , KIND = scalar-int-constant-expr ) |
//        ( type-param-value , [KIND =] scalar-int-constant-expr ) |
//        ( KIND = scalar-int-constant-expr [, LEN = type-param-value] )
TYPE_PARSER(construct<CharSelector>(Parser<LengthSelector>{}) ||
    parenthesized(construct<CharSelector>(
        "LEN =" >> typeParamValue, ", KIND =" >> scalarIntConstantExpr)) ||
    parenthesized(construct<CharSelector>(
        typeParamValue / ",", maybe("KIND ="_tok) >> scalarIntConstantExpr)) ||
    parenthesized(construct<CharSelector>(
        "KIND =" >> scalarIntConstantExpr, maybe(", LEN =" >> typeParamValue))))

// R722 length-selector -> ( [LEN =] type-param-value ) | * char-length [,]
// N.B. The trailing [,] in the production is permitted by the Standard
// only in the context of a type-declaration-stmt, but even with that
// limitation, it would seem to be unnecessary and buggy to consume the comma
// here.
TYPE_PARSER(construct<LengthSelector>(
                parenthesized(maybe("LEN ="_tok) >> typeParamValue)) ||
    construct<LengthSelector>("*" >> charLength /* / maybe(","_tok) */))

// R723 char-length -> ( type-param-value ) | digit-string
TYPE_PARSER(construct<CharLength>(parenthesized(typeParamValue)) ||
    construct<CharLength>(space >> digitString64 / spaceCheck))

// R724 char-literal-constant ->
//        [kind-param _] ' [rep-char]... ' |
//        [kind-param _] " [rep-char]... "
// "rep-char" is any non-control character.  Doubled interior quotes are
// combined.  Backslash escapes can be enabled.
// N.B. the parsing of "kind-param" takes care to not consume the '_'.
TYPE_CONTEXT_PARSER("CHARACTER literal constant"_en_US,
    construct<CharLiteralConstant>(
        kindParam / underscore, charLiteralConstantWithoutKind) ||
        construct<CharLiteralConstant>(construct<std::optional<KindParam>>(),
            space >> charLiteralConstantWithoutKind))

TYPE_CONTEXT_PARSER(
    "Hollerith"_en_US, construct<HollerithLiteralConstant>(rawHollerithLiteral))

// R725 logical-literal-constant ->
//        .TRUE. [_ kind-param] | .FALSE. [_ kind-param]
// Also accept .T. and .F. as extensions.
TYPE_PARSER(construct<LogicalLiteralConstant>(logicalTRUE,
                maybe(noSpace >> underscore >> noSpace >> kindParam)) ||
    construct<LogicalLiteralConstant>(
        logicalFALSE, maybe(noSpace >> underscore >> noSpace >> kindParam)))

// R726 derived-type-def ->
//        derived-type-stmt [type-param-def-stmt]...
//        [private-or-sequence]... [component-part]
//        [type-bound-procedure-part] end-type-stmt
// R735 component-part -> [component-def-stmt]...
TYPE_CONTEXT_PARSER("derived type definition"_en_US,
    construct<DerivedTypeDef>(statement(Parser<DerivedTypeStmt>{}),
        many(unambiguousStatement(Parser<TypeParamDefStmt>{})),
        many(statement(Parser<PrivateOrSequence>{})),
        many(inContext("component"_en_US,
            unambiguousStatement(Parser<ComponentDefStmt>{}))),
        maybe(Parser<TypeBoundProcedurePart>{}),
        statement(Parser<EndTypeStmt>{})))

// R727 derived-type-stmt ->
//        TYPE [[, type-attr-spec-list] ::] type-name [(
//        type-param-name-list )]
TYPE_CONTEXT_PARSER("TYPE statement"_en_US,
    construct<DerivedTypeStmt>(
        "TYPE" >> optionalListBeforeColons(Parser<TypeAttrSpec>{}), name,
        defaulted(parenthesized(nonemptyList(name)))))

// R728 type-attr-spec ->
//        ABSTRACT | access-spec | BIND(C) | EXTENDS ( parent-type-name )
TYPE_PARSER(construct<TypeAttrSpec>(construct<Abstract>("ABSTRACT"_tok)) ||
    construct<TypeAttrSpec>(construct<TypeAttrSpec::BindC>("BIND ( C )"_tok)) ||
    construct<TypeAttrSpec>(
        construct<TypeAttrSpec::Extends>("EXTENDS" >> parenthesized(name))) ||
    construct<TypeAttrSpec>(accessSpec))

// R729 private-or-sequence -> private-components-stmt | sequence-stmt
TYPE_PARSER(construct<PrivateOrSequence>(Parser<PrivateStmt>{}) ||
    construct<PrivateOrSequence>(Parser<SequenceStmt>{}))

// R730 end-type-stmt -> END TYPE [type-name]
TYPE_PARSER(construct<EndTypeStmt>(
    recovery("END TYPE" >> maybe(name), namedConstructEndStmtErrorRecovery)))

// R731 sequence-stmt -> SEQUENCE
TYPE_PARSER(construct<SequenceStmt>("SEQUENCE"_tok))

// R732 type-param-def-stmt ->
//        integer-type-spec , type-param-attr-spec :: type-param-decl-list
// R734 type-param-attr-spec -> KIND | LEN
constexpr auto kindOrLen{"KIND" >> pure(common::TypeParamAttr::Kind) ||
    "LEN" >> pure(common::TypeParamAttr::Len)};
TYPE_PARSER(construct<TypeParamDefStmt>(integerTypeSpec / ",", kindOrLen,
    "::" >> nonemptyList("expected type parameter declarations"_err_en_US,
                Parser<TypeParamDecl>{})))

// R733 type-param-decl -> type-param-name [= scalar-int-constant-expr]
TYPE_PARSER(construct<TypeParamDecl>(name, maybe("=" >> scalarIntConstantExpr)))

// R736 component-def-stmt -> data-component-def-stmt |
//        proc-component-def-stmt
// Accidental extension not enabled here: PGI accepts type-param-def-stmt in
// component-part of derived-type-def.
TYPE_PARSER(recovery(
    withMessage("expected component definition"_err_en_US,
        first(construct<ComponentDefStmt>(Parser<DataComponentDefStmt>{}),
            construct<ComponentDefStmt>(Parser<ProcComponentDefStmt>{}),
            construct<ComponentDefStmt>(indirect(compilerDirective)))),
    construct<ComponentDefStmt>(inStmtErrorRecovery)))

// R737 data-component-def-stmt ->
//        declaration-type-spec [[, component-attr-spec-list] ::]
//        component-decl-list
// N.B. The standard requires double colons if there's an initializer.
TYPE_PARSER(construct<DataComponentDefStmt>(declarationTypeSpec,
    optionalListBeforeColons(Parser<ComponentAttrSpec>{}),
    nonemptyList("expected component declarations"_err_en_US,
        Parser<ComponentOrFill>{})))

// R738 component-attr-spec ->
//        access-spec | ALLOCATABLE |
//        CODIMENSION lbracket coarray-spec rbracket |
//        CONTIGUOUS | DIMENSION ( component-array-spec ) | POINTER |
//        CUDA-data-attr
TYPE_PARSER(construct<ComponentAttrSpec>(accessSpec) ||
    construct<ComponentAttrSpec>(allocatable) ||
    construct<ComponentAttrSpec>("CODIMENSION" >> coarraySpec) ||
    construct<ComponentAttrSpec>(contiguous) ||
    construct<ComponentAttrSpec>("DIMENSION" >> Parser<ComponentArraySpec>{}) ||
    construct<ComponentAttrSpec>(pointer) ||
    extension<LanguageFeature::CUDA>(
        construct<ComponentAttrSpec>(Parser<common::CUDADataAttr>{})) ||
    construct<ComponentAttrSpec>(recovery(
        fail<ErrorRecovery>(
            "type parameter definitions must appear before component declarations"_err_en_US),
        kindOrLen >> construct<ErrorRecovery>())))

// R739 component-decl ->
//        component-name [( component-array-spec )]
//        [lbracket coarray-spec rbracket] [* char-length]
//        [component-initialization]
TYPE_CONTEXT_PARSER("component declaration"_en_US,
    construct<ComponentDecl>(name, maybe(Parser<ComponentArraySpec>{}),
        maybe(coarraySpec), maybe("*" >> charLength), maybe(initialization)))
// The source field of the Name will be replaced with a distinct generated name.
TYPE_CONTEXT_PARSER("%FILL item"_en_US,
    extension<LanguageFeature::DECStructures>(
        "nonstandard usage: %FILL"_port_en_US,
        construct<FillDecl>(space >> sourced("%FILL" >> construct<Name>()),
            maybe(Parser<ComponentArraySpec>{}), maybe("*" >> charLength))))
TYPE_PARSER(construct<ComponentOrFill>(Parser<ComponentDecl>{}) ||
    construct<ComponentOrFill>(Parser<FillDecl>{}))

// R740 component-array-spec ->
//        explicit-shape-spec-list | deferred-shape-spec-list
// N.B. Parenthesized here rather than around references to this production.
TYPE_PARSER(construct<ComponentArraySpec>(parenthesized(
                nonemptyList("expected explicit shape specifications"_err_en_US,
                    explicitShapeSpec))) ||
    construct<ComponentArraySpec>(parenthesized(deferredShapeSpecList)))

// R741 proc-component-def-stmt ->
//        PROCEDURE ( [proc-interface] ) , proc-component-attr-spec-list
//          :: proc-decl-list
TYPE_CONTEXT_PARSER("PROCEDURE component definition statement"_en_US,
    construct<ProcComponentDefStmt>(
        "PROCEDURE" >> parenthesized(maybe(procInterface)),
        localRecovery("expected PROCEDURE component attributes"_err_en_US,
            "," >> nonemptyList(Parser<ProcComponentAttrSpec>{}), ok),
        localRecovery("expected PROCEDURE declarations"_err_en_US,
            "::" >> nonemptyList(procDecl), SkipTo<'\n'>{})))

// R742 proc-component-attr-spec ->
//        access-spec | NOPASS | PASS [(arg-name)] | POINTER
constexpr auto noPass{construct<NoPass>("NOPASS"_tok)};
constexpr auto pass{construct<Pass>("PASS" >> maybe(parenthesized(name)))};
TYPE_PARSER(construct<ProcComponentAttrSpec>(accessSpec) ||
    construct<ProcComponentAttrSpec>(noPass) ||
    construct<ProcComponentAttrSpec>(pass) ||
    construct<ProcComponentAttrSpec>(pointer))

// R744 initial-data-target -> designator
constexpr auto initialDataTarget{indirect(designator)};

// R743 component-initialization ->
//        = constant-expr | => null-init | => initial-data-target
// R805 initialization ->
//        = constant-expr | => null-init | => initial-data-target
// Universal extension: initialization -> / data-stmt-value-list /
TYPE_PARSER(construct<Initialization>("=>" >> nullInit) ||
    construct<Initialization>("=>" >> initialDataTarget) ||
    construct<Initialization>("=" >> constantExpr) ||
    extension<LanguageFeature::SlashInitialization>(
        "nonstandard usage: /initialization/"_port_en_US,
        construct<Initialization>(
            "/" >> nonemptyList("expected values"_err_en_US,
                       indirect(Parser<DataStmtValue>{})) /
                "/")))

// R745 private-components-stmt -> PRIVATE
// R747 binding-private-stmt -> PRIVATE
TYPE_PARSER(construct<PrivateStmt>("PRIVATE"_tok))

// R746 type-bound-procedure-part ->
//        contains-stmt [binding-private-stmt] [type-bound-proc-binding]...
TYPE_CONTEXT_PARSER("type bound procedure part"_en_US,
    construct<TypeBoundProcedurePart>(statement(containsStmt),
        maybe(statement(Parser<PrivateStmt>{})),
        many(statement(Parser<TypeBoundProcBinding>{}))))

// R748 type-bound-proc-binding ->
//        type-bound-procedure-stmt | type-bound-generic-stmt |
//        final-procedure-stmt
TYPE_CONTEXT_PARSER("type bound procedure binding"_en_US,
    recovery(
        first(construct<TypeBoundProcBinding>(Parser<TypeBoundProcedureStmt>{}),
            construct<TypeBoundProcBinding>(Parser<TypeBoundGenericStmt>{}),
            construct<TypeBoundProcBinding>(Parser<FinalProcedureStmt>{})),
        construct<TypeBoundProcBinding>(
            !"END"_tok >> SkipTo<'\n'>{} >> construct<ErrorRecovery>())))

// R749 type-bound-procedure-stmt ->
//        PROCEDURE [[, bind-attr-list] ::] type-bound-proc-decl-list |
//        PROCEDURE ( interface-name ) , bind-attr-list :: binding-name-list
// The "::" is required by the standard (C768) in the first production if
// any type-bound-proc-decl has a "=>', but it's not strictly necessary to
// avoid a bad parse.
TYPE_CONTEXT_PARSER("type bound PROCEDURE statement"_en_US,
    "PROCEDURE" >>
        (construct<TypeBoundProcedureStmt>(
             construct<TypeBoundProcedureStmt::WithInterface>(
                 parenthesized(name),
                 localRecovery("expected list of binding attributes"_err_en_US,
                     "," >> nonemptyList(Parser<BindAttr>{}), ok),
                 localRecovery("expected list of binding names"_err_en_US,
                     "::" >> listOfNames, SkipTo<'\n'>{}))) ||
            construct<TypeBoundProcedureStmt>(construct<
                TypeBoundProcedureStmt::WithoutInterface>(
                pure<std::list<BindAttr>>(),
                nonemptyList(
                    "expected type bound procedure declarations"_err_en_US,
                    construct<TypeBoundProcDecl>(name,
                        maybe(extension<LanguageFeature::MissingColons>(
                            "type-bound procedure statement should have '::' if it has '=>'"_port_en_US,
                            "=>" >> name)))))) ||
            construct<TypeBoundProcedureStmt>(
                construct<TypeBoundProcedureStmt::WithoutInterface>(
                    optionalListBeforeColons(Parser<BindAttr>{}),
                    nonemptyList(
                        "expected type bound procedure declarations"_err_en_US,
                        Parser<TypeBoundProcDecl>{})))))

// R750 type-bound-proc-decl -> binding-name [=> procedure-name]
TYPE_PARSER(construct<TypeBoundProcDecl>(name, maybe("=>" >> name)))

// R751 type-bound-generic-stmt ->
//        GENERIC [, access-spec] :: generic-spec => binding-name-list
TYPE_CONTEXT_PARSER("type bound GENERIC statement"_en_US,
    construct<TypeBoundGenericStmt>("GENERIC" >> maybe("," >> accessSpec),
        "::" >> indirect(genericSpec), "=>" >> listOfNames))

// R752 bind-attr ->
//        access-spec | DEFERRED | NON_OVERRIDABLE | NOPASS | PASS [(arg-name)]
TYPE_PARSER(construct<BindAttr>(accessSpec) ||
    construct<BindAttr>(construct<BindAttr::Deferred>("DEFERRED"_tok)) ||
    construct<BindAttr>(
        construct<BindAttr::Non_Overridable>("NON_OVERRIDABLE"_tok)) ||
    construct<BindAttr>(noPass) || construct<BindAttr>(pass))

// R753 final-procedure-stmt -> FINAL [::] final-subroutine-name-list
TYPE_CONTEXT_PARSER("FINAL statement"_en_US,
    construct<FinalProcedureStmt>("FINAL" >> maybe("::"_tok) >> listOfNames))

// R754 derived-type-spec -> type-name [(type-param-spec-list)]
TYPE_PARSER(construct<DerivedTypeSpec>(name,
    defaulted(parenthesized(nonemptyList(
        "expected type parameters"_err_en_US, Parser<TypeParamSpec>{})))))

// R755 type-param-spec -> [keyword =] type-param-value
TYPE_PARSER(construct<TypeParamSpec>(maybe(keyword / "="), typeParamValue))

// R756 structure-constructor -> derived-type-spec ( [component-spec-list] )
TYPE_PARSER((construct<StructureConstructor>(derivedTypeSpec,
                 parenthesized(optionalList(Parser<ComponentSpec>{}))) ||
                // This alternative corrects misrecognition of the
                // component-spec-list as the type-param-spec-list in
                // derived-type-spec.
                construct<StructureConstructor>(
                    construct<DerivedTypeSpec>(
                        name, construct<std::list<TypeParamSpec>>()),
                    parenthesized(optionalList(Parser<ComponentSpec>{})))) /
    !"("_tok)

// R757 component-spec -> [keyword =] component-data-source
TYPE_PARSER(construct<ComponentSpec>(
    maybe(keyword / "="), Parser<ComponentDataSource>{}))

// R758 component-data-source -> expr | data-target | proc-target
TYPE_PARSER(construct<ComponentDataSource>(indirect(expr)))

// R759 enum-def ->
//        enum-def-stmt enumerator-def-stmt [enumerator-def-stmt]...
//        end-enum-stmt
TYPE_CONTEXT_PARSER("enum definition"_en_US,
    construct<EnumDef>(statement(Parser<EnumDefStmt>{}),
        some(unambiguousStatement(Parser<EnumeratorDefStmt>{})),
        statement(Parser<EndEnumStmt>{})))

// R760 enum-def-stmt -> ENUM, BIND(C)
TYPE_PARSER(construct<EnumDefStmt>("ENUM , BIND ( C )"_tok))

// R761 enumerator-def-stmt -> ENUMERATOR [::] enumerator-list
TYPE_CONTEXT_PARSER("ENUMERATOR statement"_en_US,
    construct<EnumeratorDefStmt>("ENUMERATOR" >> maybe("::"_tok) >>
        nonemptyList("expected enumerators"_err_en_US, Parser<Enumerator>{})))

// R762 enumerator -> named-constant [= scalar-int-constant-expr]
TYPE_PARSER(
    construct<Enumerator>(namedConstant, maybe("=" >> scalarIntConstantExpr)))

// R763 end-enum-stmt -> END ENUM
TYPE_PARSER(recovery("END ENUM"_tok, constructEndStmtErrorRecovery) >>
    construct<EndEnumStmt>())

// R801 type-declaration-stmt ->
//        declaration-type-spec [[, attr-spec]... ::] entity-decl-list
constexpr auto entityDeclWithoutEqInit{construct<EntityDecl>(name,
    maybe(arraySpec), maybe(coarraySpec), maybe("*" >> charLength),
    !"="_tok >> maybe(initialization))}; // old-style REAL A/0/ still works
TYPE_PARSER(
    construct<TypeDeclarationStmt>(declarationTypeSpec,
        defaulted("," >> nonemptyList(Parser<AttrSpec>{})) / "::",
        nonemptyList("expected entity declarations"_err_en_US, entityDecl)) ||
    // C806: no initializers allowed without colons ("REALA=1" is ambiguous)
    construct<TypeDeclarationStmt>(declarationTypeSpec,
        construct<std::list<AttrSpec>>(),
        nonemptyList("expected entity declarations"_err_en_US,
            entityDeclWithoutEqInit)) ||
    // PGI-only extension: comma in place of doubled colons
    extension<LanguageFeature::MissingColons>(
        "nonstandard usage: ',' in place of '::'"_port_en_US,
        construct<TypeDeclarationStmt>(declarationTypeSpec,
            defaulted("," >> nonemptyList(Parser<AttrSpec>{})),
            withMessage("expected entity declarations"_err_en_US,
                "," >> nonemptyList(entityDecl)))))

// R802 attr-spec ->
//        access-spec | ALLOCATABLE | ASYNCHRONOUS |
//        CODIMENSION lbracket coarray-spec rbracket | CONTIGUOUS |
//        DIMENSION ( array-spec ) | EXTERNAL | INTENT ( intent-spec ) |
//        INTRINSIC | language-binding-spec | OPTIONAL | PARAMETER | POINTER |
//        PROTECTED | SAVE | TARGET | VALUE | VOLATILE |
//        CUDA-data-attr
TYPE_PARSER(construct<AttrSpec>(accessSpec) ||
    construct<AttrSpec>(allocatable) ||
    construct<AttrSpec>(construct<Asynchronous>("ASYNCHRONOUS"_tok)) ||
    construct<AttrSpec>("CODIMENSION" >> coarraySpec) ||
    construct<AttrSpec>(contiguous) ||
    construct<AttrSpec>("DIMENSION" >> arraySpec) ||
    construct<AttrSpec>(construct<External>("EXTERNAL"_tok)) ||
    construct<AttrSpec>("INTENT" >> parenthesized(intentSpec)) ||
    construct<AttrSpec>(construct<Intrinsic>("INTRINSIC"_tok)) ||
    construct<AttrSpec>(languageBindingSpec) || construct<AttrSpec>(optional) ||
    construct<AttrSpec>(construct<Parameter>("PARAMETER"_tok)) ||
    construct<AttrSpec>(pointer) || construct<AttrSpec>(protectedAttr) ||
    construct<AttrSpec>(save) ||
    construct<AttrSpec>(construct<Target>("TARGET"_tok)) ||
    construct<AttrSpec>(construct<Value>("VALUE"_tok)) ||
    construct<AttrSpec>(construct<Volatile>("VOLATILE"_tok)) ||
    extension<LanguageFeature::CUDA>(
        construct<AttrSpec>(Parser<common::CUDADataAttr>{})))

// CUDA-data-attr ->
//     CONSTANT | DEVICE | MANAGED | PINNED | SHARED | TEXTURE | UNIFIED
TYPE_PARSER("CONSTANT" >> pure(common::CUDADataAttr::Constant) ||
    "DEVICE" >> pure(common::CUDADataAttr::Device) ||
    "MANAGED" >> pure(common::CUDADataAttr::Managed) ||
    "PINNED" >> pure(common::CUDADataAttr::Pinned) ||
    "SHARED" >> pure(common::CUDADataAttr::Shared) ||
    "TEXTURE" >> pure(common::CUDADataAttr::Texture) ||
    "UNIFIED" >> pure(common::CUDADataAttr::Unified))

// R804 object-name -> name
constexpr auto objectName{name};

// R803 entity-decl ->
//        object-name [( array-spec )] [lbracket coarray-spec rbracket]
//          [* char-length] [initialization] |
//        function-name [* char-length]
TYPE_PARSER(construct<EntityDecl>(objectName, maybe(arraySpec),
    maybe(coarraySpec), maybe("*" >> charLength), maybe(initialization)))

// R806 null-init -> function-reference   ... which must resolve to NULL()
TYPE_PARSER(lookAhead(name / "( )") >> construct<NullInit>(expr))

// R807 access-spec -> PUBLIC | PRIVATE
TYPE_PARSER(construct<AccessSpec>("PUBLIC" >> pure(AccessSpec::Kind::Public)) ||
    construct<AccessSpec>("PRIVATE" >> pure(AccessSpec::Kind::Private)))

// R808 language-binding-spec ->
//        BIND ( C [, NAME = scalar-default-char-constant-expr] )
// R1528 proc-language-binding-spec -> language-binding-spec
TYPE_PARSER(construct<LanguageBindingSpec>(
    "BIND ( C" >> maybe(", NAME =" >> scalarDefaultCharConstantExpr),
    (", CDEFINED" >> pure(true) || pure(false)) / ")"))

// R809 coarray-spec -> deferred-coshape-spec-list | explicit-coshape-spec
// N.B. Bracketed here rather than around references, for consistency with
// array-spec.
TYPE_PARSER(
    construct<CoarraySpec>(bracketed(Parser<DeferredCoshapeSpecList>{})) ||
    construct<CoarraySpec>(bracketed(Parser<ExplicitCoshapeSpec>{})))

// R810 deferred-coshape-spec -> :
// deferred-coshape-spec-list - just a list of colons
inline int listLength(std::list<Success> &&xs) { return xs.size(); }

TYPE_PARSER(construct<DeferredCoshapeSpecList>(
    applyFunction(listLength, nonemptyList(":"_tok))))

// R811 explicit-coshape-spec ->
//        [[lower-cobound :] upper-cobound ,]... [lower-cobound :] *
// R812 lower-cobound -> specification-expr
// R813 upper-cobound -> specification-expr
TYPE_PARSER(construct<ExplicitCoshapeSpec>(
    many(explicitShapeSpec / ","), maybe(specificationExpr / ":") / "*"))

// R815 array-spec ->
//        explicit-shape-spec-list | assumed-shape-spec-list |
//        deferred-shape-spec-list | assumed-size-spec | implied-shape-spec |
//        implied-shape-or-assumed-size-spec | assumed-rank-spec
// N.B. Parenthesized here rather than around references to avoid
// a need for forced look-ahead.
// Shape specs that could be deferred-shape-spec or assumed-shape-spec
// (e.g. '(:,:)') are parsed as the former.
TYPE_PARSER(
    construct<ArraySpec>(parenthesized(nonemptyList(explicitShapeSpec))) ||
    construct<ArraySpec>(parenthesized(deferredShapeSpecList)) ||
    construct<ArraySpec>(
        parenthesized(nonemptyList(Parser<AssumedShapeSpec>{}))) ||
    construct<ArraySpec>(parenthesized(Parser<AssumedSizeSpec>{})) ||
    construct<ArraySpec>(parenthesized(Parser<ImpliedShapeSpec>{})) ||
    construct<ArraySpec>(parenthesized(Parser<AssumedRankSpec>{})))

// R816 explicit-shape-spec -> [lower-bound :] upper-bound
// R817 lower-bound -> specification-expr
// R818 upper-bound -> specification-expr
TYPE_PARSER(construct<ExplicitShapeSpec>(
    maybe(specificationExpr / ":"), specificationExpr))

// R819 assumed-shape-spec -> [lower-bound] :
TYPE_PARSER(construct<AssumedShapeSpec>(maybe(specificationExpr) / ":"))

// R820 deferred-shape-spec -> :
// deferred-shape-spec-list - just a list of colons
TYPE_PARSER(construct<DeferredShapeSpecList>(
    applyFunction(listLength, nonemptyList(":"_tok))))

// R821 assumed-implied-spec -> [lower-bound :] *
TYPE_PARSER(construct<AssumedImpliedSpec>(maybe(specificationExpr / ":") / "*"))

// R822 assumed-size-spec -> explicit-shape-spec-list , assumed-implied-spec
TYPE_PARSER(construct<AssumedSizeSpec>(
    nonemptyList(explicitShapeSpec) / ",", assumedImpliedSpec))

// R823 implied-shape-or-assumed-size-spec -> assumed-implied-spec
// R824 implied-shape-spec -> assumed-implied-spec , assumed-implied-spec-list
// I.e., when the assumed-implied-spec-list has a single item, it constitutes an
// implied-shape-or-assumed-size-spec; otherwise, an implied-shape-spec.
TYPE_PARSER(construct<ImpliedShapeSpec>(nonemptyList(assumedImpliedSpec)))

// R825 assumed-rank-spec -> ..
TYPE_PARSER(construct<AssumedRankSpec>(".."_tok))

// R826 intent-spec -> IN | OUT | INOUT
TYPE_PARSER(construct<IntentSpec>("IN OUT" >> pure(IntentSpec::Intent::InOut) ||
    "IN" >> pure(IntentSpec::Intent::In) ||
    "OUT" >> pure(IntentSpec::Intent::Out)))

// R827 access-stmt -> access-spec [[::] access-id-list]
TYPE_PARSER(construct<AccessStmt>(accessSpec,
    defaulted(maybe("::"_tok) >>
        nonemptyList("expected names and generic specifications"_err_en_US,
            Parser<AccessId>{}))))

// R828 access-id -> access-name | generic-spec
// "access-name" is ambiguous with "generic-spec"
TYPE_PARSER(construct<AccessId>(indirect(genericSpec)))

// R829 allocatable-stmt -> ALLOCATABLE [::] allocatable-decl-list
TYPE_PARSER(construct<AllocatableStmt>("ALLOCATABLE" >> maybe("::"_tok) >>
    nonemptyList(
        "expected object declarations"_err_en_US, Parser<ObjectDecl>{})))

// R830 allocatable-decl ->
//        object-name [( array-spec )] [lbracket coarray-spec rbracket]
// R860 target-decl ->
//        object-name [( array-spec )] [lbracket coarray-spec rbracket]
TYPE_PARSER(
    construct<ObjectDecl>(objectName, maybe(arraySpec), maybe(coarraySpec)))

// R831 asynchronous-stmt -> ASYNCHRONOUS [::] object-name-list
TYPE_PARSER(construct<AsynchronousStmt>("ASYNCHRONOUS" >> maybe("::"_tok) >>
    nonemptyList("expected object names"_err_en_US, objectName)))

// R832 bind-stmt -> language-binding-spec [::] bind-entity-list
TYPE_PARSER(construct<BindStmt>(languageBindingSpec / maybe("::"_tok),
    nonemptyList("expected bind entities"_err_en_US, Parser<BindEntity>{})))

// R833 bind-entity -> entity-name | / common-block-name /
TYPE_PARSER(construct<BindEntity>(pure(BindEntity::Kind::Object), name) ||
    construct<BindEntity>("/" >> pure(BindEntity::Kind::Common), name / "/"))

// R834 codimension-stmt -> CODIMENSION [::] codimension-decl-list
TYPE_PARSER(construct<CodimensionStmt>("CODIMENSION" >> maybe("::"_tok) >>
    nonemptyList("expected codimension declarations"_err_en_US,
        Parser<CodimensionDecl>{})))

// R835 codimension-decl -> coarray-name lbracket coarray-spec rbracket
TYPE_PARSER(construct<CodimensionDecl>(name, coarraySpec))

// R836 contiguous-stmt -> CONTIGUOUS [::] object-name-list
TYPE_PARSER(construct<ContiguousStmt>("CONTIGUOUS" >> maybe("::"_tok) >>
    nonemptyList("expected object names"_err_en_US, objectName)))

// R837 data-stmt -> DATA data-stmt-set [[,] data-stmt-set]...
TYPE_CONTEXT_PARSER("DATA statement"_en_US,
    construct<DataStmt>(
        "DATA" >> nonemptySeparated(Parser<DataStmtSet>{}, maybe(","_tok))))

// R838 data-stmt-set -> data-stmt-object-list / data-stmt-value-list /
TYPE_PARSER(construct<DataStmtSet>(
    nonemptyList(
        "expected DATA statement objects"_err_en_US, Parser<DataStmtObject>{}),
    withMessage("expected DATA statement value list"_err_en_US,
        "/"_tok >> nonemptyList("expected DATA statement values"_err_en_US,
                       Parser<DataStmtValue>{})) /
        "/"))

// R839 data-stmt-object -> variable | data-implied-do
TYPE_PARSER(construct<DataStmtObject>(indirect(variable)) ||
    construct<DataStmtObject>(dataImpliedDo))

// R840 data-implied-do ->
//        ( data-i-do-object-list , [integer-type-spec ::] data-i-do-variable
//        = scalar-int-constant-expr , scalar-int-constant-expr
//        [, scalar-int-constant-expr] )
// R842 data-i-do-variable -> do-variable
TYPE_PARSER(parenthesized(construct<DataImpliedDo>(
    nonemptyList(Parser<DataIDoObject>{} / lookAhead(","_tok)) / ",",
    maybe(integerTypeSpec / "::"), loopBounds(scalarIntConstantExpr))))

// R841 data-i-do-object ->
//        array-element | scalar-structure-component | data-implied-do
TYPE_PARSER(construct<DataIDoObject>(scalar(indirect(designator))) ||
    construct<DataIDoObject>(indirect(dataImpliedDo)))

// R843 data-stmt-value -> [data-stmt-repeat *] data-stmt-constant
TYPE_PARSER(construct<DataStmtValue>(
    maybe(Parser<DataStmtRepeat>{} / "*"), Parser<DataStmtConstant>{}))

// R847 constant-subobject -> designator
// R846 int-constant-subobject -> constant-subobject
constexpr auto constantSubobject{constant(indirect(designator))};

// R844 data-stmt-repeat -> scalar-int-constant | scalar-int-constant-subobject
// R607 int-constant -> constant
// Factored into: constant -> literal-constant -> int-literal-constant
// The named-constant alternative of constant is subsumed by constant-subobject
TYPE_PARSER(construct<DataStmtRepeat>(intLiteralConstant) ||
    construct<DataStmtRepeat>(scalar(integer(constantSubobject))))

// R845 data-stmt-constant ->
//        scalar-constant | scalar-constant-subobject |
//        signed-int-literal-constant | signed-real-literal-constant |
//        null-init | initial-data-target |
//        constant-structure-constructor
// N.B. scalar-constant and scalar-constant-subobject are ambiguous with
// initial-data-target; null-init and structure-constructor are ambiguous
// in the absence of parameters and components; structure-constructor with
// components can be ambiguous with a scalar-constant-subobject.
// So we parse literal constants, designator, null-init, and
// structure-constructor, so that semantics can figure things out later
// with the symbol table.
TYPE_PARSER(sourced(first(construct<DataStmtConstant>(literalConstant),
    construct<DataStmtConstant>(signedRealLiteralConstant),
    construct<DataStmtConstant>(signedIntLiteralConstant),
    extension<LanguageFeature::SignedComplexLiteral>(
        "nonstandard usage: signed COMPLEX literal"_port_en_US,
        construct<DataStmtConstant>(Parser<SignedComplexLiteralConstant>{})),
    construct<DataStmtConstant>(nullInit),
    construct<DataStmtConstant>(indirect(designator) / !"("_tok),
    construct<DataStmtConstant>(Parser<StructureConstructor>{}))))

// R848 dimension-stmt ->
//        DIMENSION [::] array-name ( array-spec )
//        [, array-name ( array-spec )]...
TYPE_CONTEXT_PARSER("DIMENSION statement"_en_US,
    construct<DimensionStmt>("DIMENSION" >> maybe("::"_tok) >>
        nonemptyList("expected array specifications"_err_en_US,
            construct<DimensionStmt::Declaration>(name, arraySpec))))

// R849 intent-stmt -> INTENT ( intent-spec ) [::] dummy-arg-name-list
TYPE_CONTEXT_PARSER("INTENT statement"_en_US,
    construct<IntentStmt>(
        "INTENT" >> parenthesized(intentSpec) / maybe("::"_tok), listOfNames))

// R850 optional-stmt -> OPTIONAL [::] dummy-arg-name-list
TYPE_PARSER(
    construct<OptionalStmt>("OPTIONAL" >> maybe("::"_tok) >> listOfNames))

// R851 parameter-stmt -> PARAMETER ( named-constant-def-list )
// Legacy extension: omitted parentheses, no implicit typing from names
TYPE_CONTEXT_PARSER("PARAMETER statement"_en_US,
    construct<ParameterStmt>(
        "PARAMETER" >> parenthesized(nonemptyList(Parser<NamedConstantDef>{}))))
TYPE_CONTEXT_PARSER("old style PARAMETER statement"_en_US,
    extension<LanguageFeature::OldStyleParameter>(
        "nonstandard usage: PARAMETER without parentheses"_port_en_US,
        construct<OldParameterStmt>(
            "PARAMETER" >> nonemptyList(Parser<NamedConstantDef>{}))))

// R852 named-constant-def -> named-constant = constant-expr
TYPE_PARSER(construct<NamedConstantDef>(namedConstant, "=" >> constantExpr))

// R853 pointer-stmt -> POINTER [::] pointer-decl-list
TYPE_PARSER(construct<PointerStmt>("POINTER" >> maybe("::"_tok) >>
    nonemptyList(
        "expected pointer declarations"_err_en_US, Parser<PointerDecl>{})))

// R854 pointer-decl ->
//        object-name [( deferred-shape-spec-list )] | proc-entity-name
TYPE_PARSER(
    construct<PointerDecl>(name, maybe(parenthesized(deferredShapeSpecList))))

// R855 protected-stmt -> PROTECTED [::] entity-name-list
TYPE_PARSER(
    construct<ProtectedStmt>("PROTECTED" >> maybe("::"_tok) >> listOfNames))

// R856 save-stmt -> SAVE [[::] saved-entity-list]
TYPE_PARSER(construct<SaveStmt>(
    "SAVE" >> defaulted(maybe("::"_tok) >>
                  nonemptyList("expected SAVE entities"_err_en_US,
                      Parser<SavedEntity>{}))))

// R857 saved-entity -> object-name | proc-pointer-name | / common-block-name /
// R858 proc-pointer-name -> name
TYPE_PARSER(construct<SavedEntity>(pure(SavedEntity::Kind::Entity), name) ||
    construct<SavedEntity>("/" >> pure(SavedEntity::Kind::Common), name / "/"))

// R859 target-stmt -> TARGET [::] target-decl-list
TYPE_PARSER(construct<TargetStmt>("TARGET" >> maybe("::"_tok) >>
    nonemptyList("expected objects"_err_en_US, Parser<ObjectDecl>{})))

// R861 value-stmt -> VALUE [::] dummy-arg-name-list
TYPE_PARSER(construct<ValueStmt>("VALUE" >> maybe("::"_tok) >> listOfNames))

// R862 volatile-stmt -> VOLATILE [::] object-name-list
TYPE_PARSER(construct<VolatileStmt>("VOLATILE" >> maybe("::"_tok) >>
    nonemptyList("expected object names"_err_en_US, objectName)))

// R866 implicit-name-spec -> EXTERNAL | TYPE
constexpr auto implicitNameSpec{
    "EXTERNAL" >> pure(ImplicitStmt::ImplicitNoneNameSpec::External) ||
    "TYPE" >> pure(ImplicitStmt::ImplicitNoneNameSpec::Type)};

// R863 implicit-stmt ->
//        IMPLICIT implicit-spec-list |
//        IMPLICIT NONE [( [implicit-name-spec-list] )]
TYPE_CONTEXT_PARSER("IMPLICIT statement"_en_US,
    construct<ImplicitStmt>(
        "IMPLICIT" >> nonemptyList("expected IMPLICIT specifications"_err_en_US,
                          Parser<ImplicitSpec>{})) ||
        construct<ImplicitStmt>("IMPLICIT NONE"_sptok >>
            defaulted(parenthesized(optionalList(implicitNameSpec)))))

// R864 implicit-spec -> declaration-type-spec ( letter-spec-list )
// The variant form of declarationTypeSpec is meant to avoid misrecognition
// of a letter-spec as a simple parenthesized expression for kind or character
// length, e.g., PARAMETER(I=5,N=1); IMPLICIT REAL(I-N)(O-Z) vs.
// IMPLICIT REAL(I-N).  The variant form needs to attempt to reparse only
// types with optional parenthesized kind/length expressions, so derived
// type specs, DOUBLE PRECISION, and DOUBLE COMPLEX need not be considered.
constexpr auto noKindSelector{construct<std::optional<KindSelector>>()};
constexpr auto implicitSpecDeclarationTypeSpecRetry{
    construct<DeclarationTypeSpec>(first(
        construct<IntrinsicTypeSpec>(
            construct<IntegerTypeSpec>("INTEGER" >> noKindSelector)),
        construct<IntrinsicTypeSpec>(
            construct<IntrinsicTypeSpec::Real>("REAL" >> noKindSelector)),
        construct<IntrinsicTypeSpec>(
            construct<IntrinsicTypeSpec::Complex>("COMPLEX" >> noKindSelector)),
        construct<IntrinsicTypeSpec>(construct<IntrinsicTypeSpec::Character>(
            "CHARACTER" >> construct<std::optional<CharSelector>>())),
        construct<IntrinsicTypeSpec>(construct<IntrinsicTypeSpec::Logical>(
            "LOGICAL" >> noKindSelector))))};

TYPE_PARSER(construct<ImplicitSpec>(declarationTypeSpec,
                parenthesized(nonemptyList(Parser<LetterSpec>{}))) ||
    construct<ImplicitSpec>(implicitSpecDeclarationTypeSpecRetry,
        parenthesized(nonemptyList(Parser<LetterSpec>{}))))

// R865 letter-spec -> letter [- letter]
TYPE_PARSER(space >> (construct<LetterSpec>(letter, maybe("-" >> letter)) ||
                         construct<LetterSpec>(otherIdChar,
                             construct<std::optional<const char *>>())))

// R867 import-stmt ->
//        IMPORT [[::] import-name-list] |
//        IMPORT , ONLY : import-name-list | IMPORT , NONE | IMPORT , ALL
TYPE_CONTEXT_PARSER("IMPORT statement"_en_US,
    construct<ImportStmt>(
        "IMPORT , ONLY :" >> pure(common::ImportKind::Only), listOfNames) ||
        construct<ImportStmt>(
            "IMPORT , NONE" >> pure(common::ImportKind::None)) ||
        construct<ImportStmt>(
            "IMPORT , ALL" >> pure(common::ImportKind::All)) ||
        construct<ImportStmt>(
            "IMPORT" >> maybe("::"_tok) >> optionalList(name)))

// R868 namelist-stmt ->
//        NAMELIST / namelist-group-name / namelist-group-object-list
//        [[,] / namelist-group-name / namelist-group-object-list]...
// R869 namelist-group-object -> variable-name
TYPE_PARSER(construct<NamelistStmt>("NAMELIST" >>
    nonemptySeparated(
        construct<NamelistStmt::Group>("/" >> name / "/", listOfNames),
        maybe(","_tok))))

// R870 equivalence-stmt -> EQUIVALENCE equivalence-set-list
// R871 equivalence-set -> ( equivalence-object , equivalence-object-list )
TYPE_PARSER(construct<EquivalenceStmt>("EQUIVALENCE" >>
    nonemptyList(
        parenthesized(nonemptyList("expected EQUIVALENCE objects"_err_en_US,
            Parser<EquivalenceObject>{})))))

// R872 equivalence-object -> variable-name | array-element | substring
TYPE_PARSER(construct<EquivalenceObject>(indirect(designator)))

// R873 common-stmt ->
//        COMMON [/ [common-block-name] /] common-block-object-list
//        [[,] / [common-block-name] / common-block-object-list]...
TYPE_PARSER(
    construct<CommonStmt>("COMMON" >> defaulted("/" >> maybe(name) / "/"),
        nonemptyList("expected COMMON block objects"_err_en_US,
            Parser<CommonBlockObject>{}),
        many(maybe(","_tok) >>
            construct<CommonStmt::Block>("/" >> maybe(name) / "/",
                nonemptyList("expected COMMON block objects"_err_en_US,
                    Parser<CommonBlockObject>{})))))

// R874 common-block-object -> variable-name [( array-spec )]
TYPE_PARSER(construct<CommonBlockObject>(name, maybe(arraySpec)))

// R901 designator -> object-name | array-element | array-section |
//                    coindexed-named-object | complex-part-designator |
//                    structure-component | substring
// The Standard's productions for designator and its alternatives are
// ambiguous without recourse to a symbol table.  Many of the alternatives
// for designator (viz., array-element, coindexed-named-object,
// and structure-component) are all syntactically just data-ref.
// What designator boils down to is this:
//  It starts with either a name or a character literal.
//  If it starts with a character literal, it must be a substring.
//  If it starts with a name, it's a sequence of %-separated parts;
//  each part is a name, maybe a (section-subscript-list), and
//  maybe an [image-selector].
//  If it's a substring, it ends with (substring-range).
TYPE_CONTEXT_PARSER("designator"_en_US,
    sourced(construct<Designator>(substring) || construct<Designator>(dataRef)))

constexpr auto percentOrDot{"%"_tok ||
    // legacy VAX extension for RECORD field access
    extension<LanguageFeature::DECStructures>(
        "nonstandard usage: component access with '.' in place of '%'"_port_en_US,
        "."_tok / lookAhead(OldStructureComponentName{}))};

// R902 variable -> designator | function-reference
// This production appears to be left-recursive in the grammar via
//   function-reference ->  procedure-designator -> proc-component-ref ->
//     scalar-variable
// and would be so if we were to allow functions to be called via procedure
// pointer components within derived type results of other function references
// (a reasonable extension, esp. in the case of procedure pointer components
// that are NOPASS).  However, Fortran constrains the use of a variable in a
// proc-component-ref to be a data-ref without coindices (C1027).
// Some array element references will be misrecognized as function references.
constexpr auto noMoreAddressing{!"("_tok >> !"["_tok >> !percentOrDot};
TYPE_CONTEXT_PARSER("variable"_en_US,
    construct<Variable>(indirect(functionReference / noMoreAddressing)) ||
        construct<Variable>(indirect(designator)))

// R908 substring -> parent-string ( substring-range )
// R909 parent-string ->
//        scalar-variable-name | array-element | coindexed-named-object |
//        scalar-structure-component | scalar-char-literal-constant |
//        scalar-named-constant
TYPE_PARSER(
    construct<Substring>(dataRef, parenthesized(Parser<SubstringRange>{})))

TYPE_PARSER(construct<CharLiteralConstantSubstring>(
    charLiteralConstant, parenthesized(Parser<SubstringRange>{})))

TYPE_PARSER(sourced(construct<SubstringInquiry>(Parser<Substring>{}) /
    ("%LEN"_tok || "%KIND"_tok)))

// R910 substring-range -> [scalar-int-expr] : [scalar-int-expr]
TYPE_PARSER(construct<SubstringRange>(
    maybe(scalarIntExpr), ":" >> maybe(scalarIntExpr)))

// R911 data-ref -> part-ref [% part-ref]...
// R914 coindexed-named-object -> data-ref
// R917 array-element -> data-ref
TYPE_PARSER(
    construct<DataRef>(nonemptySeparated(Parser<PartRef>{}, percentOrDot)))

// R912 part-ref -> part-name [( section-subscript-list )] [image-selector]
TYPE_PARSER(construct<PartRef>(name,
    defaulted(
        parenthesized(nonemptyList(Parser<SectionSubscript>{})) / !"=>"_tok),
    maybe(Parser<ImageSelector>{})))

// R913 structure-component -> data-ref
// The final part-ref in the data-ref is not allowed to have subscripts.
TYPE_CONTEXT_PARSER("component"_en_US,
    construct<StructureComponent>(
        construct<DataRef>(some(Parser<PartRef>{} / percentOrDot)), name))

// R919 subscript -> scalar-int-expr
constexpr auto subscript{scalarIntExpr};

// R920 section-subscript -> subscript | subscript-triplet | vector-subscript
// R923 vector-subscript -> int-expr
// N.B. The distinction that needs to be made between "subscript" and
// "vector-subscript" is deferred to semantic analysis.
TYPE_PARSER(construct<SectionSubscript>(Parser<SubscriptTriplet>{}) ||
    construct<SectionSubscript>(intExpr))

// R921 subscript-triplet -> [subscript] : [subscript] [: stride]
TYPE_PARSER(construct<SubscriptTriplet>(
    maybe(subscript), ":" >> maybe(subscript), maybe(":" >> subscript)))

// R925 cosubscript -> scalar-int-expr
constexpr auto cosubscript{scalarIntExpr};

// R924 image-selector ->
//        lbracket cosubscript-list [, image-selector-spec-list] rbracket
TYPE_CONTEXT_PARSER("image selector"_en_US,
    construct<ImageSelector>(
        "[" >> nonemptyList(cosubscript / lookAhead(space / ",]"_ch)),
        defaulted("," >> nonemptyList(Parser<ImageSelectorSpec>{})) / "]"))

// R926 image-selector-spec ->
//        STAT = stat-variable | TEAM = team-value |
//        TEAM_NUMBER = scalar-int-expr
TYPE_PARSER(construct<ImageSelectorSpec>(construct<ImageSelectorSpec::Stat>(
                "STAT =" >> scalar(integer(indirect(variable))))) ||
    construct<ImageSelectorSpec>(construct<TeamValue>("TEAM =" >> teamValue)) ||
    construct<ImageSelectorSpec>(construct<ImageSelectorSpec::Team_Number>(
        "TEAM_NUMBER =" >> scalarIntExpr)))

// R927 allocate-stmt ->
//        ALLOCATE ( [type-spec ::] allocation-list [, alloc-opt-list] )
TYPE_CONTEXT_PARSER("ALLOCATE statement"_en_US,
    construct<AllocateStmt>("ALLOCATE (" >> maybe(typeSpec / "::"),
        nonemptyList(Parser<Allocation>{}),
        defaulted("," >> nonemptyList(Parser<AllocOpt>{})) / ")"))

// R928 alloc-opt ->
//        ERRMSG = errmsg-variable | MOLD = source-expr |
//        SOURCE = source-expr | STAT = stat-variable |
// (CUDA) STREAM = scalar-int-expr
//        PINNED = scalar-logical-variable
// R931 source-expr -> expr
TYPE_PARSER(construct<AllocOpt>(
                construct<AllocOpt::Mold>("MOLD =" >> indirect(expr))) ||
    construct<AllocOpt>(
        construct<AllocOpt::Source>("SOURCE =" >> indirect(expr))) ||
    construct<AllocOpt>(statOrErrmsg) ||
    extension<LanguageFeature::CUDA>(
        construct<AllocOpt>(construct<AllocOpt::Stream>(
            "STREAM =" >> indirect(scalarIntExpr))) ||
        construct<AllocOpt>(construct<AllocOpt::Pinned>(
            "PINNED =" >> indirect(scalarLogicalVariable)))))

// R929 stat-variable -> scalar-int-variable
TYPE_PARSER(construct<StatVariable>(scalar(integer(variable))))

// R932 allocation ->
//        allocate-object [( allocate-shape-spec-list )]
//        [lbracket allocate-coarray-spec rbracket]
TYPE_PARSER(construct<Allocation>(Parser<AllocateObject>{},
    defaulted(parenthesized(nonemptyList(Parser<AllocateShapeSpec>{}))),
    maybe(bracketed(Parser<AllocateCoarraySpec>{}))))

// R933 allocate-object -> variable-name | structure-component
TYPE_PARSER(construct<AllocateObject>(structureComponent) ||
    construct<AllocateObject>(name / !"="_tok))

// R934 allocate-shape-spec -> [lower-bound-expr :] upper-bound-expr
// R938 allocate-coshape-spec -> [lower-bound-expr :] upper-bound-expr
TYPE_PARSER(construct<AllocateShapeSpec>(maybe(boundExpr / ":"), boundExpr))

// R937 allocate-coarray-spec ->
//      [allocate-coshape-spec-list ,] [lower-bound-expr :] *
TYPE_PARSER(construct<AllocateCoarraySpec>(
    defaulted(nonemptyList(Parser<AllocateShapeSpec>{}) / ","),
    maybe(boundExpr / ":") / "*"))

// R939 nullify-stmt -> NULLIFY ( pointer-object-list )
TYPE_CONTEXT_PARSER("NULLIFY statement"_en_US,
    "NULLIFY" >> parenthesized(construct<NullifyStmt>(
                     nonemptyList(Parser<PointerObject>{}))))

// R940 pointer-object ->
//        variable-name | structure-component | proc-pointer-name
TYPE_PARSER(construct<PointerObject>(structureComponent) ||
    construct<PointerObject>(name))

// R941 deallocate-stmt ->
//        DEALLOCATE ( allocate-object-list [, dealloc-opt-list] )
TYPE_CONTEXT_PARSER("DEALLOCATE statement"_en_US,
    construct<DeallocateStmt>(
        "DEALLOCATE (" >> nonemptyList(Parser<AllocateObject>{}),
        defaulted("," >> nonemptyList(statOrErrmsg)) / ")"))

// R942 dealloc-opt -> STAT = stat-variable | ERRMSG = errmsg-variable
// R1165 sync-stat -> STAT = stat-variable | ERRMSG = errmsg-variable
TYPE_PARSER(construct<StatOrErrmsg>("STAT =" >> statVariable) ||
    construct<StatOrErrmsg>("ERRMSG =" >> msgVariable))

// Directives, extensions, and deprecated statements
// !DIR$ IGNORE_TKR [ [(tkrdmac...)] name ]...
// !DIR$ LOOP COUNT (n1[, n2]...)
// !DIR$ name[=value] [, name[=value]]...
// !DIR$ <anything else>
constexpr auto ignore_tkr{
    "IGNORE_TKR" >> optionalList(construct<CompilerDirective::IgnoreTKR>(
                        maybe(parenthesized(many(letter))), name))};
constexpr auto loopCount{
    "LOOP COUNT" >> construct<CompilerDirective::LoopCount>(
                        parenthesized(nonemptyList(digitString64)))};
constexpr auto assumeAligned{"ASSUME_ALIGNED" >>
    optionalList(construct<CompilerDirective::AssumeAligned>(
        indirect(designator), ":"_tok >> digitString64))};
constexpr auto vectorAlways{
    "VECTOR ALWAYS" >> construct<CompilerDirective::VectorAlways>()};
TYPE_PARSER(beginDirective >> "DIR$ "_tok >>
    sourced((construct<CompilerDirective>(ignore_tkr) ||
                construct<CompilerDirective>(loopCount) ||
                construct<CompilerDirective>(assumeAligned) ||
                construct<CompilerDirective>(vectorAlways) ||
                construct<CompilerDirective>(
                    many(construct<CompilerDirective::NameValue>(
                        name, maybe(("="_tok || ":"_tok) >> digitString64))))) /
            endOfStmt ||
        construct<CompilerDirective>(pure<CompilerDirective::Unrecognized>()) /
            SkipTo<'\n'>{}))

TYPE_PARSER(extension<LanguageFeature::CrayPointer>(
    "nonstandard usage: based POINTER"_port_en_US,
    construct<BasedPointerStmt>(
        "POINTER" >> nonemptyList("expected POINTER associations"_err_en_US,
                         construct<BasedPointer>("(" >> objectName / ",",
                             objectName, maybe(Parser<ArraySpec>{}) / ")")))))

// CUDA-attributes-stmt -> ATTRIBUTES (CUDA-data-attr) [::] name-list
TYPE_PARSER(extension<LanguageFeature::CUDA>(construct<CUDAAttributesStmt>(
    "ATTRIBUTES" >> parenthesized(Parser<common::CUDADataAttr>{}),
    defaulted(
        maybe("::"_tok) >> nonemptyList("expected names"_err_en_US, name)))))

// Subtle: A structure's name includes the surrounding slashes, which avoids
// clashes with other uses of the name in the same scope.
constexpr auto structureName{maybe(sourced("/" >> name / "/"))};

// Note that Parser<StructureStmt>{} has a mandatory list of entity-decls
// and is used only by NestedStructureStmt{}.Parse() in user-state.cpp.
TYPE_PARSER(construct<StructureStmt>("STRUCTURE" >> structureName,
    localRecovery(
        "entity declarations are required on a nested structure"_err_en_US,
        nonemptyList(entityDecl), ok)))

constexpr auto nestedStructureDef{
    CONTEXT_PARSER("nested STRUCTURE definition"_en_US,
        construct<StructureDef>(statement(NestedStructureStmt{}),
            many(Parser<StructureField>{}),
            statement(construct<StructureDef::EndStructureStmt>(
                "END STRUCTURE"_tok))))};

TYPE_PARSER(construct<StructureField>(statement(StructureComponents{})) ||
    construct<StructureField>(indirect(Parser<Union>{})) ||
    construct<StructureField>(indirect(nestedStructureDef)))

TYPE_CONTEXT_PARSER("STRUCTURE definition"_en_US,
    extension<LanguageFeature::DECStructures>(
        "nonstandard usage: STRUCTURE"_port_en_US,
        construct<StructureDef>(
            statement(construct<StructureStmt>(
                "STRUCTURE" >> structureName, optionalList(entityDecl))),
            many(Parser<StructureField>{}),
            statement(construct<StructureDef::EndStructureStmt>(
                "END STRUCTURE"_tok)))))

TYPE_CONTEXT_PARSER("UNION definition"_en_US,
    construct<Union>(statement(construct<Union::UnionStmt>("UNION"_tok)),
        many(Parser<Map>{}),
        statement(construct<Union::EndUnionStmt>("END UNION"_tok))))

TYPE_CONTEXT_PARSER("MAP definition"_en_US,
    construct<Map>(statement(construct<Map::MapStmt>("MAP"_tok)),
        many(Parser<StructureField>{}),
        statement(construct<Map::EndMapStmt>("END MAP"_tok))))

TYPE_CONTEXT_PARSER("arithmetic IF statement"_en_US,
    deprecated<LanguageFeature::ArithmeticIF>(construct<ArithmeticIfStmt>(
        "IF" >> parenthesized(expr), label / ",", label / ",", label)))

TYPE_CONTEXT_PARSER("ASSIGN statement"_en_US,
    deprecated<LanguageFeature::Assign>(
        construct<AssignStmt>("ASSIGN" >> label, "TO" >> name)))

TYPE_CONTEXT_PARSER("assigned GOTO statement"_en_US,
    deprecated<LanguageFeature::AssignedGOTO>(construct<AssignedGotoStmt>(
        "GO TO" >> name,
        defaulted(maybe(","_tok) >>
            parenthesized(nonemptyList("expected labels"_err_en_US, label))))))

TYPE_CONTEXT_PARSER("PAUSE statement"_en_US,
    deprecated<LanguageFeature::Pause>(
        construct<PauseStmt>("PAUSE" >> maybe(Parser<StopCode>{}))))

// These requirement productions are defined by the Fortran standard but never
// used directly by the grammar:
//   R620 delimiter -> ( | ) | / | [ | ] | (/ | /)
//   R1027 numeric-expr -> expr
//   R1031 int-constant-expr -> int-expr
//   R1221 dtv-type-spec -> TYPE ( derived-type-spec ) |
//           CLASS ( derived-type-spec )
//
// These requirement productions are defined and used, but need not be
// defined independently here in this file:
//   R771 lbracket -> [
//   R772 rbracket -> ]
//
// Further note that:
//   R607 int-constant -> constant
//     is used only once via R844 scalar-int-constant
//   R904 logical-variable -> variable
//     is used only via scalar-logical-variable
//   R906 default-char-variable -> variable
//     is used only via scalar-default-char-variable
//   R907 int-variable -> variable
//     is used only via scalar-int-variable
//   R915 complex-part-designator -> designator % RE | designator % IM
//     %RE and %IM are initially recognized as structure components
//   R916 type-param-inquiry -> designator % type-param-name
//     is occulted by structure component designators
//   R918 array-section ->
//        data-ref [( substring-range )] | complex-part-designator
//     is not used because parsing is not sensitive to rank
//   R1030 default-char-constant-expr -> default-char-expr
//     is only used via scalar-default-char-constant-expr
} // namespace Fortran::parser
