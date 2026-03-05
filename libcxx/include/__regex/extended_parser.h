//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___REGEX_EXTENDED_PARSER_H
#define _LIBCPP___REGEX_EXTENDED_PARSER_H

#include <__algorithm/search.h>
#include <__config>
#include <__iterator/next.h>
#include <__regex/interpreter.h>
#include <__regex/parser_common.h>
#include <__regex/regex_error.h>
#include <__utility/exchange.h>
#include <string>

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __regex::__extended {

template <class _CharT, class _Traits, class _ForwardIterator>
class __parser {
  __interpreter<_CharT, _Traits> __machine_;
  _ForwardIterator __first_;
  _ForwardIterator __last_;
  regex_constants::syntax_option_type __flags_ = {};
  uint8_t __marked_count_                      = 0;
  size_t __open_count_                         = 0;

  bool __parse_ord_char() {
    _LIBCPP_ASSERT_INTERNAL(__first_ != __last_, "Expected valid character");
    switch (*__first_) {
    case '^':
    case '.':
    case '[':
    case '$':
    case '(':
    case '|':
    case '*':
    case '+':
    case '?':
    case '{':
    case '\\':
      return false;
    case ')':
      if (__open_count_ != 0)
        return false;
      [[__fallthrough__]];
    default:
      __machine_.__push_char_matcher(*__first_, __flags_);
      ++__first_;
      return true;
    }
  }
  bool __parse_quoted_char() {
    _LIBCPP_ASSERT_INTERNAL(__first_ != __last_, "Expected valid character");
    auto __next = std::next(__first_);
    if (__next == __last_ || *__first_ != '\\')
      return false;
    switch (*__next) {
    case '^':
    case '.':
    case '*':
    case '[':
    case '$':
    case '\\':
    case '(':
    case ')':
    case '|':
    case '+':
    case '?':
    case '{':
    case '}':
      __machine_.__push_char_matcher(*__next, __flags_);
      __first_ = ++__next;
      return true;
    default:
      if (regex_constants::__get_grammar(__flags_) == regex_constants::awk) {
        __first_ = __next;
        __machine_.__push_char_matcher(__parse_awk_escape<_CharT>(__first_, __last_));
        return true;
      }
      auto __val = __machine_.__get_traits().value(*__next, 10);
      if (__val < 1 || __val > 9)
        return false;
      if (__val > __marked_count_)
        std::__throw_regex_error<regex_constants::error_backref>();
      __machine_.__push_backref_matcher(__val);
      __first_ = ++__next;
      return true;
    }
  }

  bool __parse_one_char_or_coll_elem() {
    if (__first_ == __last_)
      return false;
    if (__parse_ord_char() || __parse_quoted_char())
      return true;
    if (*__first_ == '.') {
      __machine_.__push_any_matcher();
      ++__first_;
      return true;
    }
    return __regex::__parse_bracket_expression(__machine_, __first_, __last_, __flags_);
  }

  bool __parse_dupl_symbol(size_t __expr_start) {
    if (__first_ == __last_)
      return false;
    switch (*__first_) {
    case '*': {
      ++__first_;
      __machine_.__push_n_to_m_matcher(__expr_start, 0, numeric_limits<size_t>::max());
    } break;

    case '+': {
      ++__first_;
      __machine_.__push_n_to_m_matcher(__expr_start, 1, numeric_limits<size_t>::max());
    } break;

    case '?': {
      ++__first_;
      __machine_.__push_n_to_m_matcher(__expr_start, 0, 1);
    } break;

    case '{': {
      ++__first_;
      size_t __min;
      if (auto __res = __regex::__parse_dup_count(__first_, __last_)) {
        __min = __res.__num_;
      } else {
        std::__throw_regex_error<regex_constants::error_badbrace>();
      }
      if (__first_ == __last_)
        std::__throw_regex_error<regex_constants::error_brace>();

      switch (*__first_) {
      case '}': {
        ++__first_;
        __machine_.__push_n_matcher(__expr_start, __min);
        return true;
      } break;

      case ',': {
        ++__first_;
        if (__first_ == __last_)
          std::__throw_regex_error<regex_constants::error_badbrace>();
        if (*__first_ == '}') {
          ++__first_;
          __machine_.__push_n_to_m_matcher(__expr_start, __min, numeric_limits<size_t>::max());
          return true;
        }

        size_t __max;
        if (auto __res = __regex::__parse_dup_count(__first_, __last_)) {
          __max = __res.__num_;
        } else {
          std::__throw_regex_error<regex_constants::error_brace>();
        }
        if (__first_ == __last_ || *__first_ != '}')
          std::__throw_regex_error<regex_constants::error_brace>();
        ++__first_;
        if (__max < __min)
          std::__throw_regex_error<regex_constants::error_badbrace>();
        __machine_.__push_n_to_m_matcher(__expr_start, __min, __max);
        return true;
      } break;
      default:
        std::__throw_regex_error<regex_constants::error_badbrace>();
      }
    } break;
    }
    return false;
  }

  bool __parse_expression() {
    if (__first_ == __last_)
      return false;
    size_t __expr_start = __machine_.size();
    if (!__parse_one_char_or_coll_elem()) {
      switch (*__first_) {
      case '^':
        ++__first_;
        __machine_.__push_start_anchor();
        break;
      case '$':
        ++__first_;
        __machine_.__push_end_anchor();
        break;
      case '(': {
        ++__first_;
        size_t __subexpr_num = __marked_count_;
        __marked_count_ += !(__flags_ & regex_constants::nosubs);
        ++__open_count_;
        if (!(__flags_ & regex_constants::nosubs))
          __machine_.__push_subexpression_begin(__subexpr_num);
        __parse_reg_exp(__machine_.size());
        if (__first_ == __last_ || *__first_ != ')')
          std::__throw_regex_error<regex_constants::error_paren>();
        ++__first_;
        --__open_count_;
        if (!(__flags_ & regex_constants::nosubs))
          __machine_.__push_subexpression_end(__subexpr_num);
      } break;
      default:
        return false;
      }
    }
    __parse_dupl_symbol(__expr_start);
    return true;
  }

  bool __parse_branch() {
    if (!__parse_expression())
      std::__throw_regex_error<regex_constants::__re_err_empty>();
    while (__parse_expression())
      ;
    return true;
  }

  void __parse_reg_exp(size_t __expr_start) {
    if (!__parse_branch())
      std::__throw_regex_error<regex_constants::__re_err_empty>();

    while (__first_ != __last_ && *__first_ == '|') {
      ++__first_;
      auto __expr2_start = __machine_.size();
      if (!__parse_branch())
        std::__throw_regex_error<regex_constants::__re_err_empty>();
      __machine_.__push_alternative(__expr_start, __expr2_start);
    }
  }

public:
  __parser(
      _ForwardIterator __first, _ForwardIterator __last, _Traits __traits, regex_constants::syntax_option_type __flags)
      : __machine_(__traits, true), __first_(__first), __last_(__last), __flags_(__flags) {}

  void __parse_extended() {
    __parse_reg_exp(0);
    if (__first_ != __last_)
      std::__throw_regex_error<regex_constants::__re_err_parse>();
    __machine_.__push_end_state();
  }

  void __parse_egrep() {
    auto __newline = std::find(__first_, __last_, '\n');
    if (__newline != __first_) {
      std::swap(__last_, __newline);
      __parse_extended();
      std::swap(__last_, __newline);
      if (__first_ != __newline)
        std::__throw_regex_error<regex_constants::__re_err_grammar>();
    }
    if (__first_ != __last_)
      ++__first_;
    while (__first_ != __last_) {
      __newline = std::find(__first_, __last_, '\n');
      auto __expr2_start = __machine_.size();
      if (__newline != __first_) {
        std::swap(__last_, __newline);
        __parse_extended();
        std::swap(__last_, __newline);
        if (__first_ != __newline)
          std::__throw_regex_error<regex_constants::__re_err_grammar>();
      } else {
        __machine_.__push_end_state();
      }
      __machine_.__push_alternative(0, __expr2_start);
      if (__first_ != __last_)
        ++__first_;
    }
    __machine_.__push_end_state();
  }

  __interpreter<_CharT, _Traits> __extract_interpreter() { return std::move(__machine_); }
  size_t mark_count() const { return __marked_count_; }
};
} // namespace __regex::__extended

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___REGEX_EXTENDED_PARSER_H
