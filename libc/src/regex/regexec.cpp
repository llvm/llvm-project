//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of regexec (stub).
///
//===----------------------------------------------------------------------===//

#include "src/regex/regexec.h"

#include "hdr/regex_macros.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

namespace {

constexpr size_t INF_REP = ~static_cast<size_t>(0);

struct RegexCtx {
  const char *str_begin;
  const char *str_end;
  int eflags;
};

constexpr bool is_digit(unsigned char c) { return c >= '0' && c <= '9'; }
constexpr bool is_lower(unsigned char c) { return c >= 'a' && c <= 'z'; }
constexpr bool is_upper(unsigned char c) { return c >= 'A' && c <= 'Z'; }
constexpr bool is_alpha(unsigned char c) { return is_lower(c) || is_upper(c); }
constexpr bool is_alnum(unsigned char c) { return is_alpha(c) || is_digit(c); }
constexpr bool is_word(unsigned char c) { return is_alnum(c) || c == '_'; }
constexpr bool is_blank(unsigned char c) { return c == ' ' || c == '\t'; }
constexpr bool is_space(unsigned char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
         c == '\v';
}
constexpr bool is_cntrl(unsigned char c) { return c < 0x20 || c == 0x7f; }
constexpr bool is_print(unsigned char c) { return c >= 0x20 && c < 0x7f; }
constexpr bool is_graph(unsigned char c) { return c > 0x20 && c < 0x7f; }
constexpr bool is_punct(unsigned char c) { return is_graph(c) && !is_alnum(c); }
constexpr bool is_xdigit(unsigned char c) {
  return is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

bool str_eq(const char *a, size_t n, const char *b) {
  for (size_t i = 0; i < n; ++i) {
    if (b[i] == '\0' || a[i] != b[i])
      return false;
  }
  return b[n] == '\0';
}

bool match_posix_class(const char *name, size_t len, unsigned char c) {
  if (str_eq(name, len, "alnum"))
    return is_alnum(c);
  if (str_eq(name, len, "alpha"))
    return is_alpha(c);
  if (str_eq(name, len, "blank"))
    return is_blank(c);
  if (str_eq(name, len, "cntrl"))
    return is_cntrl(c);
  if (str_eq(name, len, "digit"))
    return is_digit(c);
  if (str_eq(name, len, "graph"))
    return is_graph(c);
  if (str_eq(name, len, "lower"))
    return is_lower(c);
  if (str_eq(name, len, "print"))
    return is_print(c);
  if (str_eq(name, len, "punct"))
    return is_punct(c);
  if (str_eq(name, len, "space"))
    return is_space(c);
  if (str_eq(name, len, "upper"))
    return is_upper(c);
  if (str_eq(name, len, "xdigit"))
    return is_xdigit(c);
  return false;
}

const char *find_bracket_end(const char *p, const char *p_end) {
  const char *cur = p;
  if (cur < p_end && *cur == '^')
    ++cur;
  if (cur < p_end && *cur == ']')
    ++cur;
  while (cur < p_end) {
    if (*cur == '\\' && cur + 1 < p_end) {
      cur += 2;
      continue;
    }
    if (*cur == '[' && cur + 1 < p_end &&
        (cur[1] == ':' || cur[1] == '.' || cur[1] == '=')) {
      char delim = cur[1];
      cur += 2;
      while (cur + 1 < p_end && !(cur[0] == delim && cur[1] == ']'))
        ++cur;
      if (cur + 1 < p_end)
        cur += 2;
      continue;
    }
    if (*cur == ']')
      return cur;
    ++cur;
  }
  return p_end;
}

const char *find_group_end(const char *p, const char *p_end) {
  int depth = 1;
  const char *cur = p;
  while (cur < p_end) {
    if (*cur == '\\' && cur + 1 < p_end) {
      cur += 2;
      continue;
    }
    if (*cur == '[') {
      cur = find_bracket_end(cur + 1, p_end);
      if (cur < p_end)
        ++cur;
      continue;
    }
    if (*cur == '(') {
      ++depth;
    } else if (*cur == ')') {
      --depth;
      if (depth == 0)
        return cur;
    }
    ++cur;
  }
  return p_end;
}

const char *find_top_level_bar(const char *p, const char *p_end) {
  const char *cur = p;
  while (cur < p_end) {
    if (*cur == '\\' && cur + 1 < p_end) {
      cur += 2;
      continue;
    }
    if (*cur == '[') {
      cur = find_bracket_end(cur + 1, p_end);
      if (cur < p_end)
        ++cur;
      continue;
    }
    if (*cur == '(') {
      cur = find_group_end(cur + 1, p_end);
      if (cur < p_end)
        ++cur;
      continue;
    }
    if (*cur == '|')
      return cur;
    ++cur;
  }
  return nullptr;
}

bool match_bracket(const char *b_start, const char *b_end, char ch) {
  unsigned char c = static_cast<unsigned char>(ch);
  const char *cur = b_start;
  bool negated = false;
  if (cur < b_end && *cur == '^') {
    negated = true;
    ++cur;
  }
  bool matched = false;
  while (cur < b_end) {
    if (*cur == '[' && cur + 1 < b_end && cur[1] == ':') {
      const char *cls_start = cur + 2;
      const char *cls_end = cls_start;
      while (cls_end + 1 < b_end && !(cls_end[0] == ':' && cls_end[1] == ']')) {
        ++cls_end;
      }
      if (cls_end + 1 < b_end) {
        if (match_posix_class(cls_start,
                              static_cast<size_t>(cls_end - cls_start), c))
          matched = true;
        cur = cls_end + 2;
        continue;
      }
    }
    unsigned char c1 = static_cast<unsigned char>(*cur++);
    if (c1 == '\\' && cur < b_end)
      c1 = static_cast<unsigned char>(*cur++);
    if (cur + 1 < b_end && *cur == '-') {
      ++cur;
      unsigned char c2 = static_cast<unsigned char>(*cur++);
      if (c2 == '\\' && cur < b_end)
        c2 = static_cast<unsigned char>(*cur++);
      if (c1 <= c && c <= c2)
        matched = true;
    } else {
      if (c1 == c)
        matched = true;
    }
  }
  return matched != negated;
}

bool match_escape(char esc, char ch) {
  unsigned char c = static_cast<unsigned char>(ch);
  switch (esc) {
  case 'd':
    return is_digit(c);
  case 'D':
    return !is_digit(c);
  case 's':
    return is_space(c);
  case 'S':
    return !is_space(c);
  case 'w':
    return is_word(c);
  case 'W':
    return !is_word(c);
  default:
    return ch == esc;
  }
}

const char *parse_quantifier(const char *p, const char *p_end, size_t *min_rep,
                             size_t *max_rep) {
  *min_rep = 1;
  *max_rep = 1;
  if (p >= p_end)
    return p;
  if (*p == '*') {
    *min_rep = 0;
    *max_rep = INF_REP;
    return p + 1;
  }
  if (*p == '+') {
    *min_rep = 1;
    *max_rep = INF_REP;
    return p + 1;
  }
  if (*p == '?') {
    *min_rep = 0;
    *max_rep = 1;
    return p + 1;
  }
  if (*p == '{') {
    const char *cur = p + 1;
    if (cur >= p_end || !is_digit(static_cast<unsigned char>(*cur)))
      return p;
    size_t m = 0;
    while (cur < p_end && is_digit(static_cast<unsigned char>(*cur))) {
      m = m * 10 + static_cast<size_t>(*cur - '0');
      ++cur;
    }
    size_t n = m;
    if (cur < p_end && *cur == ',') {
      ++cur;
      if (cur < p_end && *cur == '}') {
        n = INF_REP;
      } else if (cur < p_end && is_digit(static_cast<unsigned char>(*cur))) {
        n = 0;
        while (cur < p_end && is_digit(static_cast<unsigned char>(*cur))) {
          n = n * 10 + static_cast<size_t>(*cur - '0');
          ++cur;
        }
      } else {
        return p;
      }
    }
    if (cur < p_end && *cur == '}' && m <= n) {
      *min_rep = m;
      *max_rep = n;
      return cur + 1;
    }
  }
  return p;
}

bool match_expr(const RegexCtx &ctx, const char *p, const char *p_end,
                const char *s, const char **out_end);
bool match_concat(const RegexCtx &ctx, const char *p, const char *p_end,
                  const char *s, const char **out_end);

bool match_group_quantified(const RegexCtx &ctx, const char *sub_start,
                            const char *sub_end, size_t min_rep, size_t max_rep,
                            const char *rest, const char *p_end, const char *s,
                            const char **out_end) {
  if (max_rep == 0 || sub_start == sub_end)
    return match_concat(ctx, rest, p_end, s, out_end);

  size_t next_min = min_rep > 0 ? min_rep - 1 : 0;
  size_t next_max = max_rep == INF_REP ? INF_REP : max_rep - 1;

  // Iterate over each top-level branch in (sub_start..sub_end).
  const char *branch = sub_start;
  while (true) {
    const char *bar = find_top_level_bar(branch, sub_end);
    const char *branch_end = bar ? bar : sub_end;

    const char *s_mid = nullptr;
    if (match_concat(ctx, branch, branch_end, s, &s_mid)) {
      if (s_mid > s) {
        if (match_group_quantified(ctx, sub_start, sub_end, next_min, next_max,
                                   rest, p_end, s_mid, out_end))
          return true;
      } else {
        if (match_concat(ctx, rest, p_end, s, out_end))
          return true;
      }
      // If greedy match inside the group consumed too much for `rest`, try
      // shorter match limits down to `s`.
      for (const char *s_try = s_mid; s_try > s;) {
        --s_try;
        RegexCtx sub_ctx{ctx.str_begin, s_try,
                         ctx.eflags | (s_try < ctx.str_end ? REG_NOTEOL : 0)};
        const char *s_sub_end = nullptr;
        if (match_concat(sub_ctx, branch, branch_end, s, &s_sub_end) &&
            s_sub_end == s_try) {
          if (s_try > s) {
            if (match_group_quantified(ctx, sub_start, sub_end, next_min,
                                       next_max, rest, p_end, s_try, out_end))
              return true;
          } else {
            if (match_concat(ctx, rest, p_end, s, out_end))
              return true;
          }
        }
      }
    }

    if (!bar)
      break;
    branch = bar + 1;
  }

  if (min_rep == 0)
    return match_concat(ctx, rest, p_end, s, out_end);

  return false;
}

bool match_concat(const RegexCtx &ctx, const char *p, const char *p_end,
                  const char *s, const char **out_end) {
  if (p >= p_end) {
    *out_end = s;
    return true;
  }

  // Handle anchors (^ and $).
  if (*p == '^' || *p == '$') {
    char anchor = *p;
    size_t min_rep = 1, max_rep = 1;
    const char *rest = parse_quantifier(p + 1, p_end, &min_rep, &max_rep);
    if (min_rep > 0) {
      if (anchor == '^') {
        if (s != ctx.str_begin || (ctx.eflags & REG_NOTBOL) != 0)
          return false;
      } else {
        if (s != ctx.str_end || (ctx.eflags & REG_NOTEOL) != 0)
          return false;
      }
    }
    return match_concat(ctx, rest, p_end, s, out_end);
  }

  // Handle parenthesized group (...).
  if (*p == '(') {
    const char *g_end = find_group_end(p + 1, p_end);
    const char *after_group = (g_end < p_end) ? g_end + 1 : p_end;
    size_t min_rep = 1, max_rep = 1;
    const char *rest = parse_quantifier(after_group, p_end, &min_rep, &max_rep);
    return match_group_quantified(ctx, p + 1, g_end, min_rep, max_rep, rest,
                                  p_end, s, out_end);
  }

  // Single-character atom: bracket [...], escape \c, wildcard ., or literal.
  enum AtomKind { ATOM_LITERAL, ATOM_DOT, ATOM_ESCAPE, ATOM_BRACKET };
  AtomKind kind = ATOM_LITERAL;
  char lit_or_esc = *p;
  const char *b_start = nullptr;
  const char *b_end = nullptr;
  const char *after_atom = p + 1;

  if (*p == '.') {
    kind = ATOM_DOT;
  } else if (*p == '\\' && p + 1 < p_end) {
    kind = ATOM_ESCAPE;
    lit_or_esc = p[1];
    after_atom = p + 2;
  } else if (*p == '[') {
    kind = ATOM_BRACKET;
    b_start = p + 1;
    b_end = find_bracket_end(b_start, p_end);
    after_atom = (b_end < p_end) ? b_end + 1 : p_end;
  }

  size_t min_rep = 1, max_rep = 1;
  const char *rest = parse_quantifier(after_atom, p_end, &min_rep, &max_rep);

  auto matches_one = [&](char ch) -> bool {
    switch (kind) {
    case ATOM_DOT:
      return true;
    case ATOM_ESCAPE:
      return match_escape(lit_or_esc, ch);
    case ATOM_BRACKET:
      return match_bracket(b_start, b_end, ch);
    case ATOM_LITERAL:
      return ch == lit_or_esc;
    }
    return false;
  };

  size_t count = 0;
  while (count < max_rep && (s + count) < ctx.str_end &&
         matches_one(s[count])) {
    ++count;
  }

  if (count < min_rep)
    return false;

  for (size_t i = count;; --i) {
    if (match_concat(ctx, rest, p_end, s + i, out_end))
      return true;
    if (i == min_rep)
      break;
  }
  return false;
}

bool match_expr(const RegexCtx &ctx, const char *p, const char *p_end,
                const char *s, const char **out_end) {
  const char *branch = p;
  while (true) {
    const char *bar = find_top_level_bar(branch, p_end);
    const char *branch_end = bar ? bar : p_end;
    if (match_concat(ctx, branch, branch_end, s, out_end))
      return true;
    if (!bar)
      break;
    branch = bar + 1;
  }
  return false;
}

} // namespace

LLVM_LIBC_FUNCTION(int, regexec,
                   (const regex_t *__restrict preg,
                    const char *__restrict string, size_t nmatch,
                    regmatch_t *__restrict pmatch, int eflags)) {
  const char *pattern = static_cast<const char *>(preg->__internal);
  if (!pattern || !string)
    return REG_NOMATCH;

  const char *p_end = pattern;
  while (*p_end != '\0')
    ++p_end;

  const char *str_end = string;
  while (*str_end != '\0')
    ++str_end;

  RegexCtx ctx{string, str_end, eflags};

  for (const char *s = string;; ++s) {
    const char *match_end = nullptr;
    if (match_expr(ctx, pattern, p_end, s, &match_end)) {
      if (nmatch > 0 && pmatch != nullptr) {
        pmatch[0].rm_so = static_cast<regoff_t>(s - string);
        pmatch[0].rm_eo = static_cast<regoff_t>(match_end - string);
        for (size_t i = 1; i < nmatch; ++i) {
          pmatch[i].rm_so = -1;
          pmatch[i].rm_eo = -1;
        }
      }
      return 0;
    }
    if (s == str_end)
      break;
  }

  return REG_NOMATCH;
}

} // namespace LIBC_NAMESPACE_DECL
