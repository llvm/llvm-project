//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Regular expression parser implementation.
///
//===----------------------------------------------------------------------===//

#include "src/__support/regex/regex_parser.h"

#include "hdr/regex_macros.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include "src/__support/regex/regex_ast.h"
#include "src/__support/regex/regex_expr_pool.h"

namespace LIBC_NAMESPACE_DECL {
namespace regex {

namespace {

// Returns true for backslash escape sequences that represent backreferences,
// boundary assertions, or character-class shorthands rather than escaped
// literal characters.
constexpr bool is_unimplemented_ere_escape(char c) {
  switch (c) {
  case '1':
  case '2':
  case '3':
  case '4':
  case '5':
  case '6':
  case '7':
  case '8':
  case '9':
  case 'b':
  case 'B':
  case '<':
  case '>':
  case 'A':
  case 'Z':
  case '`':
  case '\'':
  case 'w':
  case 'W':
  case 's':
  case 'S':
  case 'd':
  case 'D':
    return true;
  default:
    return false;
  }
}

} // namespace

ErrorOr<ExprId> parse_ere(cpp::string_view pattern, [[maybe_unused]] int cflags,
                          ExprPool &pool) {
  ErrorOr<ExprId> cur = pool.empty_str();
  if (!cur.has_value())
    return cur;

  cpp::string_view rem = pattern;
  while (!rem.empty()) {
    char c = rem.front();
    rem.remove_prefix(1);
    switch (c) {
    case '\\': {
      if (rem.empty())
        return Error(REG_EESCAPE);
      c = rem.front();
      rem.remove_prefix(1);
      if (is_unimplemented_ere_escape(c))
        return Error(REG_BADPAT);
      break;
    }
    // TODO: Wire ERE metacharacters as each operator is implemented in
    // subsequent patches.
    case '.':
    case '^':
    case '$':
    case '*':
    case '+':
    case '?':
    case '(':
    case ')':
    case '[':
    case '{':
    case '|':
      return Error(REG_BADPAT);
    default:
      break;
    }

    ErrorOr<ExprId> lit = pool.make_lit(c);
    if (!lit.has_value())
      return lit;
    cur = pool.make_concat(*cur, *lit);
    if (!cur.has_value())
      return cur;
  }
  return cur;
}

} // namespace regex
} // namespace LIBC_NAMESPACE_DECL
