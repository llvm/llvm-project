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
#include "src/__support/CPP/array.h"
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

// Parses a single ERE atom and an optional trailing '*' quantifier.
ErrorOr<ExprId> parse_quantified_atom(cpp::string_view &rem, int cflags,
                                      ExprPool &pool) {
  char c = rem.front();
  rem.remove_prefix(1);
  ErrorOr<ExprId> atom = INVALID_EXPR_ID;
  bool can_repeat = true;
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
    can_repeat = false;
    break;
  case '$':
    atom = pool.make_anchor(ANCHOR_EOL);
    can_repeat = false;
    break;
  // Valid postfix '*' is consumed after a repeatable atom below; reaching a
  // quantifier at the start of an atom means it has no preceding repeatable
  // atom (e.g. "*a", "^*", or doubled "a**").
  case '*':
  case '+':
  case '?':
    return Error(REG_BADRPT);
  case '{':
    if (!rem.empty() && rem.front() >= '0' && rem.front() <= '9')
      return Error(REG_BADRPT);
    return Error(REG_BADPAT);
  // TODO: Wire ERE metacharacters as each operator is implemented in
  // subsequent patches.
  case '(':
  case ')':
  case '[':
  case '|':
    return Error(REG_BADPAT);
  default:
    atom = pool.make_lit(c);
    break;
  }

  if (!atom.has_value() || !can_repeat)
    return atom;
  if (rem.starts_with('*')) {
    rem.remove_prefix(1);
    return pool.make_star(*atom);
  }
  if (rem.starts_with('+') || rem.starts_with('?') || rem.starts_with('{'))
    return Error(REG_BADPAT);
  return atom;
}

// Folds concatenated atoms right-to-left in 64-atom (256-byte) stack chunks so
// right-associative make_concat stays O(1) without orphaned prefixes and
// recursion is bounded to <= 157 frames across MAX_NODE_LIMIT.
ErrorOr<ExprId> parse_concat(cpp::string_view &rem, int cflags, ExprPool &pool,
                             size_t total_atoms) {
  constexpr size_t CHUNK_SIZE = 64;
  cpp::array<ExprId, CHUNK_SIZE> atoms{};
  size_t count = 0;
  while (!rem.empty() && count < CHUNK_SIZE) {
    if (total_atoms >= ExprPool::MAX_NODE_LIMIT)
      return Error(REG_ESPACE);
    ErrorOr<ExprId> atom = parse_quantified_atom(rem, cflags, pool);
    if (!atom.has_value())
      return atom;
    atoms[count] = *atom;
    ++count;
    ++total_atoms;
  }

  ErrorOr<ExprId> acc = rem.empty()
                            ? pool.empty_str()
                            : parse_concat(rem, cflags, pool, total_atoms);
  if (!acc.has_value())
    return acc;

  for (size_t i = count; i > 0; --i) {
    acc = pool.make_concat(atoms[i - 1], *acc);
    if (!acc.has_value())
      return acc;
  }
  return acc;
}

} // namespace

ErrorOr<ExprId> parse_ere(cpp::string_view pattern, int cflags,
                          ExprPool &pool) {
  return parse_concat(pattern, cflags, pool, 0);
}

} // namespace regex
} // namespace LIBC_NAMESPACE_DECL
