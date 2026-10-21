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

ErrorOr<ExprId> parse_ere(cpp::string_view pattern, int cflags,
                          ExprPool &pool) {
  ErrorOr<ExprId> cur = pool.empty_str();
  if (!cur.has_value())
    return cur;

  cpp::string_view rem = pattern;
  while (!rem.empty()) {
    char c = rem.front();
    rem.remove_prefix(1);
    ErrorOr<ExprId> atom = INVALID_EXPR_ID;
    switch (c) {
    case '\\': {
      if (rem.empty())
        return Error(REG_EESCAPE);
      c = rem.front();
      rem.remove_prefix(1);
      if (is_unimplemented_ere_escape(c))
        return Error(REG_BADPAT);
      atom = pool.make_lit(c);
      break;
    }
    case '.': {
      ByteSet bs;
      if (cflags & REG_NEWLINE) {
        bs.set_range(0, '\n' - 1);
        bs.set_range('\n' + 1, BYTE_SET_BITS - 1);
      } else {
        bs.flip();
      }
      atom = pool.make_byte_set(bs);
      break;
    }
    case '^':
      atom = pool.make_anchor(ANCHOR_BOL);
      break;
    case '$':
      atom = pool.make_anchor(ANCHOR_EOL);
      break;
    // TODO: Wire ERE metacharacters as each operator is implemented in
    // subsequent patches.
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
      atom = pool.make_lit(c);
      break;
    }

    if (!atom.has_value())
      return atom;
    cur = pool.make_concat(*cur, *atom);
    if (!cur.has_value())
      return cur;
  }
  return cur;
}

} // namespace regex
} // namespace LIBC_NAMESPACE_DECL
