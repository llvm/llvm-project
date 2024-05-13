//===-- lib/Parser/stmt-parser.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_PARSER_STMT_PARSER_H_
#define FORTRAN_PARSER_STMT_PARSER_H_

// Basic parsing of statements.

#include "basic-parsers.h"
#include "token-parsers.h"

namespace Fortran::parser {

// statement(p) parses Statement<P> for some statement type P that is the
// result type of the argument parser p, while also handling labels and
// end-of-statement markers.

// R611 label -> digit [digit]...
constexpr auto label{space >> digitString64 / spaceCheck};

template <typename PA>
inline constexpr auto unterminatedStatement(const PA &p) {
  return skipStuffBeforeStatement >>
      sourced(construct<Statement<typename PA::resultType>>(
          maybe(label), space >> p));
}

constexpr auto endOfLine{
    "\n"_ch >> ok || fail("expected end of line"_err_en_US)};

constexpr auto semicolons{";"_ch >> skipMany(";"_tok) / space / maybe("\n"_ch)};
constexpr auto endOfStmt{
    space >> withMessage("expected end of statement"_err_en_US,
                 semicolons || endOfLine)};
constexpr auto forceEndOfStmt{recovery(endOfStmt, SkipPast<'\n'>{})};

template <typename PA> inline constexpr auto statement(const PA &p) {
  return unterminatedStatement(p) / endOfStmt;
}

// unlabeledStatement() is basically statement() for those few situations
// in Fortran where a statement cannot have a label.
template <typename PA> inline constexpr auto unlabeledStatement(const PA &p) {
  return space >>
      sourced(construct<UnlabeledStatement<typename PA::resultType>>(p));
}

// This unambiguousStatement() variant of statement() provides better error
// recovery for contexts containing statements that might have trailing
// garbage, but it must be used only when no instance of the statement in
// question could also be a legal prefix of some other statement that might
// be valid at that point.  It only makes sense to use this within "some()"
// or "many()" so as to not end the list of statements.
template <typename PA> inline constexpr auto unambiguousStatement(const PA &p) {
  return unterminatedStatement(p) / forceEndOfStmt;
}

constexpr auto ignoredStatementPrefix{
    skipStuffBeforeStatement >> maybe(label) >> maybe(name / ":") >> space};

// Error recovery within a statement() call: skip *to* the end of the line,
// unless at an END or CONTAINS statement.
constexpr auto inStmtErrorRecovery{!"END"_tok >> !"CONTAINS"_tok >>
    SkipTo<'\n'>{} >> construct<ErrorRecovery>()};

// Error recovery within statement sequences: skip *past* the end of the line,
// but not over an END or CONTAINS statement.
constexpr auto skipStmtErrorRecovery{!"END"_tok >> !"CONTAINS"_tok >>
    SkipPast<'\n'>{} >> construct<ErrorRecovery>()};

// Error recovery across statements: skip the line, unless it looks
// like it might end the containing construct.
constexpr auto stmtErrorRecoveryStart{ignoredStatementPrefix};
constexpr auto skipBadLine{SkipPast<'\n'>{} >> construct<ErrorRecovery>()};
constexpr auto executionPartErrorRecovery{stmtErrorRecoveryStart >>
    !"END"_tok >> !"CONTAINS"_tok >> !"ELSE"_tok >> !"CASE"_tok >>
    !"TYPE IS"_tok >> !"CLASS"_tok >> !"RANK"_tok >>
    !("!$ACC "_sptok >> "END"_tok) >>
    !("!$OMP "_sptok >> ("END"_tok || "SECTION"_id)) >> skipBadLine};

// END statement error recovery
constexpr auto missingOptionalName{pure<std::optional<Name>>()};
constexpr auto noNameEnd{"END" >> missingOptionalName};
constexpr auto atEndOfStmt{space >>
    withMessage("expected end of statement"_err_en_US, lookAhead(";\n"_ch))};
constexpr auto bareEnd{noNameEnd / recovery(atEndOfStmt, SkipTo<'\n'>{})};

// For unrecognizable construct END statements.  Be sure to not consume
// a program unit's END statement.
constexpr auto progUnitEndStmt{
    "END" >> (lookAhead("\n"_ch) || "SUBROUTINE"_tok || "FUNCTION"_tok ||
                 "PROCEDURE"_tok || "MODULE"_tok || "SUBMODULE"_tok ||
                 "PROGRAM"_tok || "BLOCK DATA"_tok)};
constexpr auto constructEndStmtErrorRecovery{
    !progUnitEndStmt >> ("END"_tok >> SkipTo<'\n'>{} || ok)};
constexpr auto namedConstructEndStmtErrorRecovery{
    constructEndStmtErrorRecovery >> missingOptionalName};

constexpr auto progUnitEndStmtErrorRecovery{
    (many(!"END"_tok >> SkipPast<'\n'>{}) >>
        ("END"_tok >> SkipTo<'\n'>{} || consumedAllInput)) >>
    missingOptionalName};

constexpr auto beginDirective{skipStuffBeforeStatement >> "!"_ch};
constexpr auto endDirective{space >> endOfLine};

} // namespace Fortran::parser
#endif // FORTRAN_PARSER_STMT_PARSER_H_
