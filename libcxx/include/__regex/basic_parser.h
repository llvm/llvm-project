//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___REGEX_BASIC_PARSER_H
#define _LIBCPP___REGEX_BASIC_PARSER_H

#include <__algorithm/search.h>
#include <__config>
#include <__iterator/next.h>
#include <__regex/interpreter.h>
#include <__regex/parser_common.h>
#include <__regex/regex_error.h>
#include <__utility/exchange.h>

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __regex::__basic::inline __preserve_none {

template <class _CharT, class _Traits, class _ForwardIterator>
class __parser {
  __interpreter<_CharT, _Traits> __machine_;
  _ForwardIterator __first_;
  _ForwardIterator __last_;
  regex_constants::syntax_option_type __flags_ = regex_constants::syntax_option_type();
  uint8_t __marked_count_                      = 0;

  bool __parse_ord_char(_ForwardIterator& __first, _ForwardIterator __last) {
    _LIBCPP_ASSERT_INTERNAL(__first != __last, "Expected parseable character");
    auto __next = std::next(__first);
    if (__next == __last && *__first == '$')
      return false;
    if (*__first == '.' || *__first == '\\' || *__first == '[')
      return false;
    __machine_.__push_char_matcher(*__first, __flags_);
    __first = __next;
    return true;
  }

  bool __parse_quoted_char() {
    _LIBCPP_ASSERT_INTERNAL(__first_ != __last_, "Expected parseable character");
    auto __next = std::next(__first_);
    if (__next == __last_)
      return false;
    if (*__first_ != '\\')
      return false;
    switch (*__next) {
    case '^':
    case '.':
    case '*':
    case '[':
    case '$':
    case '\\':
      __machine_.__push_char_matcher(*__next, __flags_);
      __first_ = ++__next;
      return true;
    default:
      return false;
    }
  }

  bool __parse_one_char_or_coll_elem() {
    if (__first_ == __last_)
      return false;

    if (__parse_ord_char(__first_, __last_))
      return true;
    if (__parse_quoted_char())
      return true;

    if (*__first_ == '.') {
      __machine_.__push_any_matcher();
      ++__first_;
      return true;
    } else {
      return __regex::__parse_bracket_expression(__machine_, __first_, __last_, __flags_);
    }
  }

  bool __parse_backref() {
    if (__first_ == __last_)
      return false;

    auto __next = std::next(__first_);
    if (__next == __last_)
      return false;
    if (*__first_ != '\\' || *__next < '0' || *__next > '9')
      return false;
    if (*__next == '0') {
      __machine_.__push_char_matcher('\0');
      __first_ = ++__next;
      return true;
    }
    auto __val = *__next - '0';
    if (__val > __marked_count_)
      std::__throw_regex_error<regex_constants::error_backref>();
    __machine_.__push_backref_matcher(__val, __flags_);
    __first_ = ++__next;
    return true;
  }

  bool __parse_escaped_character(_CharT __char) {
    if (__first_ == __last_)
      return false;
    auto __next = std::next(__first_);
    if (__next == __last_)
      return false;
    if (*__first_ != '\\' || *__next != __char)
      return false;
    __first_ = ++__next;
    return true;
  }

  bool __parse_nondupl() {
    if (__parse_one_char_or_coll_elem())
      return true;

    if (__parse_escaped_character('(')) {
      size_t __subexpr_num = __marked_count_++;
      __machine_.__push_subexpression_begin(__subexpr_num);
      __parse_re_expression();
      if (!__parse_escaped_character(')'))
        std::__throw_regex_error<regex_constants::error_paren>();
      __machine_.__push_subexpression_end(__subexpr_num);
      return true;
    } else {
      return __parse_backref();
    }
  }

  bool __parse_dupl_symbol(size_t __expr_start) {
    if (__first_ == __last_)
      return false;

    if (*__first_ == '*') {
      __machine_.__push_n_to_m_matcher(__expr_start, 0, numeric_limits<size_t>::max());
      ++__first_;
      return true;
    }

    if (!__parse_escaped_character('{'))
      return false;

    size_t __min;
    if (auto __res = __regex::__parse_dup_count(__first_, __last_)) {
      __min = __res.__num_;
    } else {
      std::__throw_regex_error<regex_constants::error_badbrace>();
    }

    if (__first_ == __last_)
      std::__throw_regex_error<regex_constants::error_brace>();
    if (*__first_ == ',') {
      ++__first_;
      size_t __max;
      if (auto __res = __regex::__parse_dup_count(__first_, __last_)) {
        __max = __res.__num_;
      } else {
        __max = numeric_limits<size_t>::max();
      }

      if (__max < __min || !__parse_escaped_character('}'))
        std::__throw_regex_error<regex_constants::error_badbrace>();
      __machine_.__push_n_to_m_matcher(__expr_start, __min, __max);
      return true;
    }

    if (!__parse_escaped_character('}'))
      std::__throw_regex_error<regex_constants::error_brace>();
    __machine_.__push_n_matcher(__expr_start, __min);
    return true;
  }

  bool __parse_simple_re() {
    if (__first_ == __last_)
      return false;

    auto __expr_start = __machine_.size();
    if (__parse_nondupl()) {
      __parse_dupl_symbol(__expr_start);
      return true;
    }
    return false;
  }

  void __parse_re_expression() {
    while (__parse_simple_re())
      ;
  }

  void __parse_basic_expr() {
    if (__first_ == __last_)
      return;

    if (*__first_ == '^') {
      __machine_.__push_start_anchor();
      ++__first_;
      if (__first_ == __last_)
        return;
    }

    __parse_re_expression();

    if (__first_ != __last_) {
      auto __next = std::next(__first_);
      if (__next == __last_ && *__first_ == '$') {
        __machine_.__push_end_anchor();
        __first_ = __next;
      }
    }

    if (__first_ != __last_)
      std::__throw_regex_error<regex_constants::__re_err_empty>();
  }

public:
  __parser(
      _ForwardIterator __first, _ForwardIterator __last, _Traits __traits, regex_constants::syntax_option_type __flags)
      : __machine_(__traits), __first_(__first), __last_(__last), __flags_(__flags) {}

  void __parse_basic() {
    __parse_basic_expr();
    __machine_.__push_end_state();
  }

  void __parse_grep() {
    auto __newline = std::find(__first_, __last_, '\n');
    if (__newline != __first_) {
      std::swap(__last_, __newline);
      __parse_basic_expr();
      std::swap(__last_, __newline);
      if (__first_ != __newline)
        std::__throw_regex_error<regex_constants::__re_err_grammar>();
    }
    if (__first_ != __last_)
      ++__first_;
    while (__first_ != __last_) {
      __newline = std::find(__first_, __last_, '\n');
      std::swap(__last_, __newline);
      __parse_basic_expr();
      std::swap(__last_, __newline);
      if (__first_ != __newline)
        std::__throw_regex_error<regex_constants::__re_err_grammar>();
      __machine_.__push_alternative(0, __machine_.size());
      if (__first_ != __last_)
        ++__first_;
    }
    __machine_.__push_end_state();
  }

  __interpreter<_CharT, _Traits> __extract_interpreter() { return std::move(__machine_); }
  size_t mark_count() const { return __marked_count_; }
};
} // namespace __regex::__basic::inline __preserve_none

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___REGEX_BASIC_PARSER_H
