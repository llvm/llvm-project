//===-- lib/Parser/expr-parsers.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Per-type parsers for expressions.

#include "expr-parsers.h"
#include "basic-parsers.h"
#include "debug-parser.h"
#include "misc-parsers.h"
#include "stmt-parser.h"
#include "token-parsers.h"
#include "type-parser-implementation.h"
#include "flang/Parser/characters.h"
#include "flang/Parser/parse-tree.h"

namespace Fortran::parser {

// R764 boz-literal-constant -> binary-constant | octal-constant | hex-constant
// R765 binary-constant -> B ' digit [digit]... ' | B " digit [digit]... "
// R766 octal-constant -> O ' digit [digit]... ' | O " digit [digit]... "
// R767 hex-constant ->
//        Z ' hex-digit [hex-digit]... ' | Z " hex-digit [hex-digit]... "
// extension: X accepted for Z
// extension: BOZX suffix accepted
TYPE_PARSER(construct<BOZLiteralConstant>(BOZLiteral{}))

// R769 array-constructor -> (/ ac-spec /) | lbracket ac-spec rbracket
TYPE_CONTEXT_PARSER("array constructor"_en_US,
    construct<ArrayConstructor>(
        "(/" >> Parser<AcSpec>{} / "/)" || bracketed(Parser<AcSpec>{})))

// R770 ac-spec -> type-spec :: | [type-spec ::] ac-value-list
TYPE_PARSER(construct<AcSpec>(maybe(typeSpec / "::"),
                nonemptyList("expected array constructor values"_err_en_US,
                    Parser<AcValue>{})) ||
    construct<AcSpec>(typeSpec / "::"))

// R773 ac-value -> expr | ac-implied-do
TYPE_PARSER(
    // PGI/Intel extension: accept triplets in array constructors
    extension<LanguageFeature::TripletInArrayConstructor>(
        "nonstandard usage: triplet in array constructor"_port_en_US,
        construct<AcValue>(construct<AcValue::Triplet>(scalarIntExpr,
            ":" >> scalarIntExpr, maybe(":" >> scalarIntExpr)))) ||
    construct<AcValue>(indirect(expr)) ||
    construct<AcValue>(indirect(Parser<AcImpliedDo>{})))

// R774 ac-implied-do -> ( ac-value-list , ac-implied-do-control )
TYPE_PARSER(parenthesized(
    construct<AcImpliedDo>(nonemptyList(Parser<AcValue>{} / lookAhead(","_tok)),
        "," >> Parser<AcImpliedDoControl>{})))

// R775 ac-implied-do-control ->
//        [integer-type-spec ::] ac-do-variable = scalar-int-expr ,
//        scalar-int-expr [, scalar-int-expr]
// R776 ac-do-variable -> do-variable
TYPE_PARSER(construct<AcImpliedDoControl>(
    maybe(integerTypeSpec / "::"), loopBounds(scalarIntExpr)))

// R1001 primary ->
//         literal-constant | designator | array-constructor |
//         structure-constructor | function-reference | type-param-inquiry |
//         type-param-name | ( expr )
// type-param-inquiry is parsed as a structure component, except for
// substring%KIND/LEN
constexpr auto primary{instrumented("primary"_en_US,
    first(construct<Expr>(indirect(Parser<CharLiteralConstantSubstring>{})),
        construct<Expr>(literalConstant),
        construct<Expr>(construct<Expr::Parentheses>(parenthesized(expr))),
        construct<Expr>(indirect(functionReference) / !"("_tok / !"%"_tok),
        construct<Expr>(designator / !"("_tok / !"%"_tok),
        construct<Expr>(indirect(Parser<SubstringInquiry>{})), // %LEN or %KIND
        construct<Expr>(Parser<StructureConstructor>{}),
        construct<Expr>(Parser<ArrayConstructor>{}),
        // PGI/XLF extension: COMPLEX constructor (x,y)
        extension<LanguageFeature::ComplexConstructor>(
            "nonstandard usage: generalized COMPLEX constructor"_port_en_US,
            construct<Expr>(parenthesized(
                construct<Expr::ComplexConstructor>(expr, "," >> expr)))),
        extension<LanguageFeature::PercentLOC>(
            "nonstandard usage: %LOC"_port_en_US,
            construct<Expr>("%LOC" >> parenthesized(construct<Expr::PercentLoc>(
                                          indirect(variable)))))))};

// R1002 level-1-expr -> [defined-unary-op] primary
// TODO: Reasonable extension: permit multiple defined-unary-ops
constexpr auto level1Expr{sourced(
    first(primary, // must come before define op to resolve .TRUE._8 ambiguity
        construct<Expr>(construct<Expr::DefinedUnary>(definedOpName, primary)),
        extension<LanguageFeature::SignedPrimary>(
            "nonstandard usage: signed primary"_port_en_US,
            construct<Expr>(construct<Expr::UnaryPlus>("+" >> primary))),
        extension<LanguageFeature::SignedPrimary>(
            "nonstandard usage: signed primary"_port_en_US,
            construct<Expr>(construct<Expr::Negate>("-" >> primary)))))};

// R1004 mult-operand -> level-1-expr [power-op mult-operand]
// R1007 power-op -> **
// Exponentiation (**) is Fortran's only right-associative binary operation.
struct MultOperand {
  using resultType = Expr;
  constexpr MultOperand() {}
  static inline std::optional<Expr> Parse(ParseState &);
};

static constexpr auto multOperand{sourced(MultOperand{})};

inline std::optional<Expr> MultOperand::Parse(ParseState &state) {
  std::optional<Expr> result{level1Expr.Parse(state)};
  if (result) {
    static constexpr auto op{attempt("**"_tok)};
    if (op.Parse(state)) {
      std::function<Expr(Expr &&)> power{[&result](Expr &&right) {
        return Expr{Expr::Power(std::move(result).value(), std::move(right))};
      }};
      return applyLambda(power, multOperand).Parse(state); // right-recursive
    }
  }
  return result;
}

// R1005 add-operand -> [add-operand mult-op] mult-operand
// R1008 mult-op -> * | /
// The left recursion in the grammar is implemented iteratively.
struct AddOperand {
  using resultType = Expr;
  constexpr AddOperand() {}
  static inline std::optional<Expr> Parse(ParseState &state) {
    std::optional<Expr> result{multOperand.Parse(state)};
    if (result) {
      auto source{result->source};
      std::function<Expr(Expr &&)> multiply{[&result](Expr &&right) {
        return Expr{
            Expr::Multiply(std::move(result).value(), std::move(right))};
      }};
      std::function<Expr(Expr &&)> divide{[&result](Expr &&right) {
        return Expr{Expr::Divide(std::move(result).value(), std::move(right))};
      }};
      auto more{attempt(sourced("*" >> applyLambda(multiply, multOperand) ||
          "/" >> applyLambda(divide, multOperand)))};
      while (std::optional<Expr> next{more.Parse(state)}) {
        result = std::move(next);
        result->source.ExtendToCover(source);
      }
    }
    return result;
  }
};
constexpr AddOperand addOperand;

// R1006 level-2-expr -> [[level-2-expr] add-op] add-operand
// R1009 add-op -> + | -
// These are left-recursive productions, implemented iteratively.
// Note that standard Fortran admits a unary + or - to appear only here,
// by means of a missing first operand; e.g., 2*-3 is valid in C but not
// standard Fortran.  We accept unary + and - to appear before any primary
// as an extension.
struct Level2Expr {
  using resultType = Expr;
  constexpr Level2Expr() {}
  static inline std::optional<Expr> Parse(ParseState &state) {
    static constexpr auto unary{
        sourced(
            construct<Expr>(construct<Expr::UnaryPlus>("+" >> addOperand)) ||
            construct<Expr>(construct<Expr::Negate>("-" >> addOperand))) ||
        addOperand};
    std::optional<Expr> result{unary.Parse(state)};
    if (result) {
      auto source{result->source};
      std::function<Expr(Expr &&)> add{[&result](Expr &&right) {
        return Expr{Expr::Add(std::move(result).value(), std::move(right))};
      }};
      std::function<Expr(Expr &&)> subtract{[&result](Expr &&right) {
        return Expr{
            Expr::Subtract(std::move(result).value(), std::move(right))};
      }};
      auto more{attempt(sourced("+" >> applyLambda(add, addOperand) ||
          "-" >> applyLambda(subtract, addOperand)))};
      while (std::optional<Expr> next{more.Parse(state)}) {
        result = std::move(next);
        result->source.ExtendToCover(source);
      }
    }
    return result;
  }
};
constexpr Level2Expr level2Expr;

// R1010 level-3-expr -> [level-3-expr concat-op] level-2-expr
// R1011 concat-op -> //
// Concatenation (//) is left-associative for parsing performance, although
// one would never notice if it were right-associated.
struct Level3Expr {
  using resultType = Expr;
  constexpr Level3Expr() {}
  static inline std::optional<Expr> Parse(ParseState &state) {
    std::optional<Expr> result{level2Expr.Parse(state)};
    if (result) {
      auto source{result->source};
      std::function<Expr(Expr &&)> concat{[&result](Expr &&right) {
        return Expr{Expr::Concat(std::move(result).value(), std::move(right))};
      }};
      auto more{attempt(sourced("//" >> applyLambda(concat, level2Expr)))};
      while (std::optional<Expr> next{more.Parse(state)}) {
        result = std::move(next);
        result->source.ExtendToCover(source);
      }
    }
    return result;
  }
};
constexpr Level3Expr level3Expr;

// R1012 level-4-expr -> [level-3-expr rel-op] level-3-expr
// R1013 rel-op ->
//         .EQ. | .NE. | .LT. | .LE. | .GT. | .GE. |
//          == | /= | < | <= | > | >=  @ | <>
// N.B. relations are not recursive (i.e., LOGICAL is not ordered)
struct Level4Expr {
  using resultType = Expr;
  constexpr Level4Expr() {}
  static inline std::optional<Expr> Parse(ParseState &state) {
    std::optional<Expr> result{level3Expr.Parse(state)};
    if (result) {
      auto source{result->source};
      std::function<Expr(Expr &&)> lt{[&result](Expr &&right) {
        return Expr{Expr::LT(std::move(result).value(), std::move(right))};
      }};
      std::function<Expr(Expr &&)> le{[&result](Expr &&right) {
        return Expr{Expr::LE(std::move(result).value(), std::move(right))};
      }};
      std::function<Expr(Expr &&)> eq{[&result](Expr &&right) {
        return Expr{Expr::EQ(std::move(result).value(), std::move(right))};
      }};
      std::function<Expr(Expr &&)> ne{[&result](Expr &&right) {
        return Expr{Expr::NE(std::move(result).value(), std::move(right))};
      }};
      std::function<Expr(Expr &&)> ge{[&result](Expr &&right) {
        return Expr{Expr::GE(std::move(result).value(), std::move(right))};
      }};
      std::function<Expr(Expr &&)> gt{[&result](Expr &&right) {
        return Expr{Expr::GT(std::move(result).value(), std::move(right))};
      }};
      auto more{attempt(
          sourced((".LT."_tok || "<"_tok) >> applyLambda(lt, level3Expr) ||
              (".LE."_tok || "<="_tok) >> applyLambda(le, level3Expr) ||
              (".EQ."_tok || "=="_tok) >> applyLambda(eq, level3Expr) ||
              (".NE."_tok || "/="_tok ||
                  extension<LanguageFeature::AlternativeNE>(
                      "nonstandard usage: <> for /= or .NE."_port_en_US,
                      "<>"_tok /* PGI/Cray extension; Cray also has .LG. */)) >>
                  applyLambda(ne, level3Expr) ||
              (".GE."_tok || ">="_tok) >> applyLambda(ge, level3Expr) ||
              (".GT."_tok || ">"_tok) >> applyLambda(gt, level3Expr)))};
      if (std::optional<Expr> next{more.Parse(state)}) {
        next->source.ExtendToCover(source);
        return next;
      }
    }
    return result;
  }
};
constexpr Level4Expr level4Expr;

// R1014 and-operand -> [not-op] level-4-expr
// R1018 not-op -> .NOT.
// N.B. Fortran's .NOT. binds less tightly than its comparison operators do.
// PGI/Intel extension: accept multiple .NOT. operators
struct AndOperand {
  using resultType = Expr;
  constexpr AndOperand() {}
  static inline std::optional<Expr> Parse(ParseState &);
};
constexpr AndOperand andOperand;

// Match a logical operator or, optionally, its abbreviation.
inline constexpr auto logicalOp(const char *op, const char *abbrev) {
  return TokenStringMatch{op} ||
      extension<LanguageFeature::LogicalAbbreviations>(
          "nonstandard usage: abbreviated LOGICAL operator"_port_en_US,
          TokenStringMatch{abbrev});
}

inline std::optional<Expr> AndOperand::Parse(ParseState &state) {
  static constexpr auto notOp{attempt(logicalOp(".NOT.", ".N.") >> andOperand)};
  if (std::optional<Expr> negation{notOp.Parse(state)}) {
    return Expr{Expr::NOT{std::move(*negation)}};
  } else {
    return level4Expr.Parse(state);
  }
}

// R1015 or-operand -> [or-operand and-op] and-operand
// R1019 and-op -> .AND.
// .AND. is left-associative
struct OrOperand {
  using resultType = Expr;
  constexpr OrOperand() {}
  static inline std::optional<Expr> Parse(ParseState &state) {
    static constexpr auto operand{sourced(andOperand)};
    std::optional<Expr> result{operand.Parse(state)};
    if (result) {
      auto source{result->source};
      std::function<Expr(Expr &&)> logicalAnd{[&result](Expr &&right) {
        return Expr{Expr::AND(std::move(result).value(), std::move(right))};
      }};
      auto more{attempt(sourced(
          logicalOp(".AND.", ".A.") >> applyLambda(logicalAnd, andOperand)))};
      while (std::optional<Expr> next{more.Parse(state)}) {
        result = std::move(next);
        result->source.ExtendToCover(source);
      }
    }
    return result;
  }
};
constexpr OrOperand orOperand;

// R1016 equiv-operand -> [equiv-operand or-op] or-operand
// R1020 or-op -> .OR.
// .OR. is left-associative
struct EquivOperand {
  using resultType = Expr;
  constexpr EquivOperand() {}
  static inline std::optional<Expr> Parse(ParseState &state) {
    std::optional<Expr> result{orOperand.Parse(state)};
    if (result) {
      auto source{result->source};
      std::function<Expr(Expr &&)> logicalOr{[&result](Expr &&right) {
        return Expr{Expr::OR(std::move(result).value(), std::move(right))};
      }};
      auto more{attempt(sourced(
          logicalOp(".OR.", ".O.") >> applyLambda(logicalOr, orOperand)))};
      while (std::optional<Expr> next{more.Parse(state)}) {
        result = std::move(next);
        result->source.ExtendToCover(source);
      }
    }
    return result;
  }
};
constexpr EquivOperand equivOperand;

// R1017 level-5-expr -> [level-5-expr equiv-op] equiv-operand
// R1021 equiv-op -> .EQV. | .NEQV.
// Logical equivalence is left-associative.
// Extension: .XOR. as synonym for .NEQV.
struct Level5Expr {
  using resultType = Expr;
  constexpr Level5Expr() {}
  static inline std::optional<Expr> Parse(ParseState &state) {
    std::optional<Expr> result{equivOperand.Parse(state)};
    if (result) {
      auto source{result->source};
      std::function<Expr(Expr &&)> eqv{[&result](Expr &&right) {
        return Expr{Expr::EQV(std::move(result).value(), std::move(right))};
      }};
      std::function<Expr(Expr &&)> neqv{[&result](Expr &&right) {
        return Expr{Expr::NEQV(std::move(result).value(), std::move(right))};
      }};
      auto more{attempt(sourced(".EQV." >> applyLambda(eqv, equivOperand) ||
          (".NEQV."_tok ||
              extension<LanguageFeature::XOROperator>(
                  "nonstandard usage: .XOR./.X. spelling of .NEQV."_port_en_US,
                  logicalOp(".XOR.", ".X."))) >>
              applyLambda(neqv, equivOperand)))};
      while (std::optional<Expr> next{more.Parse(state)}) {
        result = std::move(next);
        result->source.ExtendToCover(source);
      }
    }
    return result;
  }
};
constexpr Level5Expr level5Expr;

// R1022 expr -> [expr defined-binary-op] level-5-expr
// Defined binary operators associate leftwards.
template <> std::optional<Expr> Parser<Expr>::Parse(ParseState &state) {
  std::optional<Expr> result{level5Expr.Parse(state)};
  if (result) {
    auto source{result->source};
    std::function<Expr(DefinedOpName &&, Expr &&)> defBinOp{
        [&result](DefinedOpName &&op, Expr &&right) {
          return Expr{Expr::DefinedBinary(
              std::move(op), std::move(result).value(), std::move(right))};
        }};
    auto more{attempt(
        sourced(applyLambda<Expr>(defBinOp, definedOpName, level5Expr)))};
    while (std::optional<Expr> next{more.Parse(state)}) {
      result = std::move(next);
      result->source.ExtendToCover(source);
    }
  }
  return result;
}

// R1003 defined-unary-op -> . letter [letter]... .
// R1023 defined-binary-op -> . letter [letter]... .
// R1414 local-defined-operator -> defined-unary-op | defined-binary-op
// R1415 use-defined-operator -> defined-unary-op | defined-binary-op
// C1003 A defined operator must be distinct from logical literal constants
// and intrinsic operator names; this is handled by attempting their parses
// first, and by name resolution on their definitions, for best errors.
// N.B. The name of the operator is captured with the dots around it.
constexpr auto definedOpNameChar{letter ||
    extension<LanguageFeature::PunctuationInNames>(
        "nonstandard usage: non-alphabetic character in defined operator"_port_en_US,
        "$@"_ch)};
TYPE_PARSER(
    space >> construct<DefinedOpName>(sourced("."_ch >>
                 some(definedOpNameChar) >> construct<Name>() / "."_ch)))

// R1028 specification-expr -> scalar-int-expr
TYPE_PARSER(construct<SpecificationExpr>(scalarIntExpr))

// R1032 assignment-stmt -> variable = expr
TYPE_CONTEXT_PARSER("assignment statement"_en_US,
    construct<AssignmentStmt>(variable / "=", expr))

// R1033 pointer-assignment-stmt ->
//         data-pointer-object [( bounds-spec-list )] => data-target |
//         data-pointer-object ( bounds-remapping-list ) => data-target |
//         proc-pointer-object => proc-target
// R1034 data-pointer-object ->
//         variable-name | scalar-variable % data-pointer-component-name
//   C1022 a scalar-variable shall be a data-ref
//   C1024 a data-pointer-object shall not be a coindexed object
// R1038 proc-pointer-object -> proc-pointer-name | proc-component-ref
//
// A distinction can't be made at the time of the initial parse between
// data-pointer-object and proc-pointer-object, or between data-target
// and proc-target.
TYPE_CONTEXT_PARSER("pointer assignment statement"_en_US,
    construct<PointerAssignmentStmt>(dataRef,
        parenthesized(nonemptyList(Parser<BoundsRemapping>{})), "=>" >> expr) ||
        construct<PointerAssignmentStmt>(dataRef,
            defaulted(parenthesized(nonemptyList(Parser<BoundsSpec>{}))),
            "=>" >> expr))

// R1035 bounds-spec -> lower-bound-expr :
TYPE_PARSER(construct<BoundsSpec>(boundExpr / ":"))

// R1036 bounds-remapping -> lower-bound-expr : upper-bound-expr
TYPE_PARSER(construct<BoundsRemapping>(boundExpr / ":", boundExpr))

// R1039 proc-component-ref -> scalar-variable % procedure-component-name
//   C1027 the scalar-variable must be a data-ref without coindices.
TYPE_PARSER(construct<ProcComponentRef>(structureComponent))

// R1041 where-stmt -> WHERE ( mask-expr ) where-assignment-stmt
// R1045 where-assignment-stmt -> assignment-stmt
// R1046 mask-expr -> logical-expr
TYPE_CONTEXT_PARSER("WHERE statement"_en_US,
    construct<WhereStmt>("WHERE" >> parenthesized(logicalExpr), assignmentStmt))

// R1042 where-construct ->
//         where-construct-stmt [where-body-construct]...
//         [masked-elsewhere-stmt [where-body-construct]...]...
//         [elsewhere-stmt [where-body-construct]...] end-where-stmt
TYPE_CONTEXT_PARSER("WHERE construct"_en_US,
    construct<WhereConstruct>(statement(Parser<WhereConstructStmt>{}),
        many(whereBodyConstruct),
        many(construct<WhereConstruct::MaskedElsewhere>(
            statement(Parser<MaskedElsewhereStmt>{}),
            many(whereBodyConstruct))),
        maybe(construct<WhereConstruct::Elsewhere>(
            statement(Parser<ElsewhereStmt>{}), many(whereBodyConstruct))),
        statement(Parser<EndWhereStmt>{})))

// R1043 where-construct-stmt -> [where-construct-name :] WHERE ( mask-expr )
TYPE_CONTEXT_PARSER("WHERE construct statement"_en_US,
    construct<WhereConstructStmt>(
        maybe(name / ":"), "WHERE" >> parenthesized(logicalExpr)))

// R1044 where-body-construct ->
//         where-assignment-stmt | where-stmt | where-construct
TYPE_PARSER(construct<WhereBodyConstruct>(statement(assignmentStmt)) ||
    construct<WhereBodyConstruct>(statement(whereStmt)) ||
    construct<WhereBodyConstruct>(indirect(whereConstruct)))

// R1047 masked-elsewhere-stmt ->
//         ELSEWHERE ( mask-expr ) [where-construct-name]
TYPE_CONTEXT_PARSER("masked ELSEWHERE statement"_en_US,
    construct<MaskedElsewhereStmt>(
        "ELSE WHERE" >> parenthesized(logicalExpr), maybe(name)))

// R1048 elsewhere-stmt -> ELSEWHERE [where-construct-name]
TYPE_CONTEXT_PARSER("ELSEWHERE statement"_en_US,
    construct<ElsewhereStmt>("ELSE WHERE" >> maybe(name)))

// R1049 end-where-stmt -> ENDWHERE [where-construct-name]
TYPE_CONTEXT_PARSER("END WHERE statement"_en_US,
    construct<EndWhereStmt>(
        recovery("END WHERE" >> maybe(name), endStmtErrorRecovery)))

// R1050 forall-construct ->
//         forall-construct-stmt [forall-body-construct]... end-forall-stmt
TYPE_CONTEXT_PARSER("FORALL construct"_en_US,
    construct<ForallConstruct>(statement(Parser<ForallConstructStmt>{}),
        many(Parser<ForallBodyConstruct>{}),
        statement(Parser<EndForallStmt>{})))

// R1051 forall-construct-stmt ->
//         [forall-construct-name :] FORALL concurrent-header
TYPE_CONTEXT_PARSER("FORALL construct statement"_en_US,
    construct<ForallConstructStmt>(
        maybe(name / ":"), "FORALL" >> indirect(concurrentHeader)))

// R1052 forall-body-construct ->
//         forall-assignment-stmt | where-stmt | where-construct |
//         forall-construct | forall-stmt
TYPE_PARSER(construct<ForallBodyConstruct>(statement(forallAssignmentStmt)) ||
    construct<ForallBodyConstruct>(statement(whereStmt)) ||
    construct<ForallBodyConstruct>(whereConstruct) ||
    construct<ForallBodyConstruct>(indirect(forallConstruct)) ||
    construct<ForallBodyConstruct>(statement(forallStmt)))

// R1053 forall-assignment-stmt -> assignment-stmt | pointer-assignment-stmt
TYPE_PARSER(construct<ForallAssignmentStmt>(assignmentStmt) ||
    construct<ForallAssignmentStmt>(pointerAssignmentStmt))

// R1054 end-forall-stmt -> END FORALL [forall-construct-name]
TYPE_CONTEXT_PARSER("END FORALL statement"_en_US,
    construct<EndForallStmt>(
        recovery("END FORALL" >> maybe(name), endStmtErrorRecovery)))

// R1055 forall-stmt -> FORALL concurrent-header forall-assignment-stmt
TYPE_CONTEXT_PARSER("FORALL statement"_en_US,
    construct<ForallStmt>("FORALL" >> indirect(concurrentHeader),
        unlabeledStatement(forallAssignmentStmt)))
} // namespace Fortran::parser
