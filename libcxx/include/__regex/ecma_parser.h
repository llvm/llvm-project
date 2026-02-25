//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___REGEX_ECMA_PARSER_H
#define _LIBCPP___REGEX_ECMA_PARSER_H

#include <__config>
#include <__regex/regex_error.h>
#include <__regex/state_machine.h>

#define _LIBCPP_PARSER_FUNCTION [[clang::preserve_none]]

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __regex::__ecma {
template <class _ForwardIterator>
_LIBCPP_PARSER_FUNCTION bool
__parse_disjunction(__state_machine& __machine, _ForwardIterator& __first, _ForwardIterator __last);

template <class _ForwardIterator>
_LIBCPP_PARSER_FUNCTION bool
__parse_assertion(__state_machine& __machine, _ForwardIterator& __first, _ForwardIterator __last) {
  if (__first == __last)
    return false;
  switch (*__first) {
  case '^': {
    __machine.push_back(__state::__start_anchor);
    ++__first;
    return true;
  }
  case '$': {
    __machine.push_back(__state::__end_anchor);
    ++__first;
    return true;
  }
  case '\\': {
    auto __next = std::next(__first);
    if (__next == __last)
      return false;
    if (auto __char = *__next; __char == 'b' || __char == 'B') {
      __machine.push_back(__char == 'b' ? __state::__match_word_boundary : __state::__match_no_word_boundary);
      __first = ++__next;
      return true;
    }
    return false;
  }
  case '(': {
    auto __next = std::next(__first);
    if (__next == __last || *++__next != '?' || __next == __last)
      return false;
    auto __char = *__next;
    if (__char != '=' && __char != '!')
      return false;
    __machine.push_back(__char == '=' ? __state::__positive_lookahead_start : __state::__negative_lookahead_start);
    if (!__ecma::__parse_disjunction(__machine, __first, __last))
      return false;
    __machine.push_back(__state::__lookahead_end);
    if (__first == __last || *__first++ != ')')
      std::__throw_regex_error<regex_constants::error_paren>();
    return true;
  }
  }
  return false;
}

template <class _ForwardIterator>
_LIBCPP_PARSER_FUNCTION bool
__parse_decimal_escape(__state_machine& __machine, _ForwardIterator& __first, _ForwardIterator __last) {
  if (*__first == '0') {
    __regex::__match_char(__machine, '\0');
    ++__first;
    return true;
  }

  if ('1' <= *__first && *__first <= '9') {
    unsigned __value = *__first - '0';
    for (++__first; __first != __last && '0' <= *__first && *__first <= '9'; ++__first) {
      if (__value >= numeric_limits<unsigned>::max() / 10)
        std::__throw_regex_error<regex_constants::error_backref>();
      __value = __value * 10 + *__first - '0';
    }
    if (__value == 0)
      std::__throw_regex_error<regex_constants::error_backref>();
    __machine.push_back(__state::__match_backref);
    __regex::__write_uleb(__machine, __value);
  }
  return true;
}

template <class _ForwardIterator>
_LIBCPP_PARSER_FUNCTION bool
__parse_character_class_escape(__state_machine& __machine, _ForwardIterator& __first, _ForwardIterator __last) {
  if (__first == __last)
    return false;
  switch (*__first) {
  case 'd':
    __machine.push_back(__state::__match_digits);
    break;
  case 'D':
    __machine.push_back(__state::__match_no_digits);
    break;
  case 's':
    __machine.push_back(__state::__match_whitespace);
    break;
  case 'S':
    __machine.push_back(__state::__match_no_whitespace);
    break;
  case 'w':
    __machine.push_back(__state::__match_word_character);
    break;
  case 'W':
    __machine.push_back(__state::__match_no_word_character);
    break;
  default:
    return false;
  }
  ++__first;
  return true;
}

template <class _ForwardIterator>
_LIBCPP_PARSER_FUNCTION bool
__parse_character_escape(__state_machine& __machine, _ForwardIterator& __first, _ForwardIterator __last) {
  if (__first == __last)
    return false;
  switch (*__first) {
  case 'f':
    __regex::__match_char(__machine, '\f');
    break;
  case 'n':
    __regex::__match_char(__machine, '\n');
    break;
  case 'r':
    __regex::__match_char(__machine, '\r');
    break;
  case 't':
    __regex::__match_char(__machine, '\t');
    break;
  case 'v':
    __regex::__match_char(__machine, '\v');
    break;

  case 'c': {
    auto __next = std::next(__first);
    if (__next == __last)
      std::__throw_regex_error<regex_constants::error_escape>();
    if (('A' <= *__next && *__next <= 'Z') || ('a' <= *__next && *__next <= 'z')) {
      __regex::__match_char(__machine, *__next % 32);
      __first += ++__next;
    } else {
      std::__throw_regex_error<regex_constants::error_escape>();
    }
  }
  }
  ++__first;
  return true;
}

template <class _ForwardIterator>
_LIBCPP_PARSER_FUNCTION bool
__parse_atom_escape(__state_machine& __machine, _ForwardIterator& __first, _ForwardIterator __last) {
  _LIBCPP_ASSERT_INTERNAL(*__first == '\\', "Parsing escape character without an escape start?");
  auto __next = std::next(__first);
  if (__next == __last)
    std::__throw_regex_error<regex_constants::error_escape>();

  if (__ecma::__parse_decimal_escape(__machine, __next, __last) ||
      __ecma::__parse_character_class_escape(__machine, __next, __last)) {
    __first = __next;
    return true;
  }
  return false;
}

template <class _ForwardIterator>
_LIBCPP_PARSER_FUNCTION bool
__parse_atom(__state_machine& __machine, _ForwardIterator& __first, _ForwardIterator __last) {
  if (__first == __last)
    return false;

  switch (*__first) {
  case '.': {
    __machine.push_back(__state::__match_any);
    ++__first;
    return true;
  }
  case '\\':
    return __ecma::__parse_atom_escape(__machine, __first, __last);
  }

  return false;
}

template <class _ForwardIterator>
_LIBCPP_PARSER_FUNCTION bool
__parse_term(__state_machine& __machine, _ForwardIterator& __first, _ForwardIterator __last) {
  return __ecma::__parse_assertion(__machine, __first, __last) || __ecma::__parse_atom(__machine, __first, __last);
}

template <class _ForwardIterator>
_LIBCPP_PARSER_FUNCTION void
__parse_alternative(__state_machine& __machine, _ForwardIterator& __first, _ForwardIterator __last) {
  while (true) {
    __ecma::__parse_term(__machine, __first, __last);
  }
}

template <class _ForwardIterator>
_LIBCPP_PARSER_FUNCTION bool
__parse_disjunction(__state_machine& __machine, _ForwardIterator& __first, _ForwardIterator __last) {
  while (__first != __last) {
    __ecma::__parse_alternative(__machine, __first, __last);
  }
  return true;
}

template <class _ForwardIterator>
_LIBCPP_PARSER_FUNCTION void
__parse_pattern(__state_machine& __machine, _ForwardIterator& __first, _ForwardIterator __last) {
  __ecma::__parse_disjunction(__machine, __first, __last);
}

template <class _ForwardIterator>
__state_machine __parse(_ForwardIterator __first, _ForwardIterator __last) {
  __state_machine __machine;

  __ecma::__parse_pattern(__machine, __first, __last);

  return __machine;
}
} // namespace __regex::__ecma

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___REGEX_ECMA_PARSER_H
