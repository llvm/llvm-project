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
#include <__regex/interpreter.h>
#include <__regex/regex_error.h>

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __regex::__ecma {

template <class _CharT, class _Traits, class _ForwardIterator>
class __parser {
  __interpreter<_CharT, _Traits> __machine_;
  _ForwardIterator __first_;
  _ForwardIterator __last_;
  regex_constants::syntax_option_type __flags_ = {};
  uint8_t __marked_count_                      = 0;
  size_t __open_count_                         = 0;

  bool __parse_assertion() {
    switch (*__first_) {
    case '^':
      if (__flags_ & regex_constants::multiline)
        __machine_.__push_multiline_start_anchor();
      else
        __machine_.__push_start_anchor();
      ++__first_;
      return true;

    case '$':
      if (__flags_ & regex_constants::multiline)
        __machine_.__push_multiline_end_anchor();
      else
        __machine_.__push_end_anchor();
      ++__first_;
      return true;

    case '\\': {
      auto __next = std::next(__first_);
      if (__next == __last_)
        return false;
      switch (*__next) {
      case 'b':
        __machine_.__push_word_boundary();
        break;

      case 'B':
        __machine_.__push_no_word_boundary();
        break;

      default:
        return false;
      }
      __first_ = ++__next;
      return true;
    }

    case '(': {
      auto __next = std::next(__first_);
      if (__next == __last_ || *__next != '?' || ++__next == __last_)
        return false;
      switch (*__next) {
      case '=':
      case '!': {
        bool __is_positive    = *__next == '=';
        auto __lookahead_info = __machine_.__start_lookahead();
        auto __marked_count   = std::exchange(__marked_count_, 0);
        __first_              = ++__next;
        __parse_exp();
        if (__first_ == __last_ || *__first_ != ')')
          std::__throw_regex_error<regex_constants::error_paren>();
        ++__first_;
        __machine_.__push_lookahead(__is_positive, __lookahead_info, __marked_count, __marked_count_);
        __marked_count_ += __marked_count;
        return true;
      }

      default:
        return false;
      }
    }

    default:
      return false;
    }
  }

  bool __parse_pattern_character() {
    switch (*__first_) {
    case '^':
    case '$':
    case '\\':
    case '.':
    case '*':
    case '+':
    case '?':
    case '(':
    case ')':
    case '[':
    case ']':
    case '{':
    case '}':
    case '|':
      return false;
    default:
      __machine_.__push_char_matcher(*__first_++);
      return true;
    }
  }

  bool __parse_decimal_escape() {
    if (*__first_ == '0') {
      __machine_.__push_char_matcher('\0');
      ++__first_;
      return true;
    }

    if (*__first_ < '1' || *__first_ > '9')
      return false;

    size_t __val = *__first_ - '0';

    for (++__first_; __first_ != __last_ && *__first_ >= '0' && *__first_ <= '0'; ++__first_) {
      if (__val >= numeric_limits<size_t>::max() / 10)
        std::__throw_regex_error<regex_constants::error_backref>();
      __val = 10 * __val + *__first_ - '0';
    }
    if (__val == 0 || __val > __marked_count_)
      std::__throw_regex_error<regex_constants::error_backref>();
    __machine_.__push_backref_matcher(__val);
    return true;
  }

  bool __parse_character_class_escape() {
    __bracket_expr<_CharT, _Traits> __expr;
    switch (*__first_) {
    case 'd': {
      ++__first_;
      __expr.__mask_ = ctype_base::digit;
      __machine_.__push_bracket_expr(false, __expr);
      return true;
    }

    case 'D': {
      ++__first_;
      __expr.__mask_ = ctype_base::digit;
      __machine_.__push_bracket_expr(true, __expr);
      return true;
    }

    case 's': {
      ++__first_;
      __expr.__mask_ = ctype_base::space;
      __machine_.__push_bracket_expr(false, __expr);
      return true;
    }

    case 'S': {
      ++__first_;
      __expr.__mask_ = ctype_base::space;
      __machine_.__push_bracket_expr(true, __expr);
      return true;
    }

    case 'w': {
      ++__first_;
      __expr.__mask_ = ctype_base::alnum;
      __machine_.__push_bracket_expr(false, __expr);
      return true;
    }

    case 'W': {
      ++__first_;
      __expr.__mask_ = ctype_base::alnum;
      __machine_.__push_bracket_expr(true, __expr);
      return true;
    }
    }
    return false;
  }

  bool __parse_atom() {
    switch (*__first_) {
    case '.':
      __machine_.__push_any_except_newline_matcher();
      ++__first_;
      return true;

    case '\\': {
      ++__first_;
      if (__first_ == __last_)
        std::__throw_regex_error<regex_constants::error_escape>();

      if (__parse_decimal_escape())
        return true;

      if (__parse_character_class_escape())
        return true;

      __machine_.__push_char_matcher(__regex::__parse_character_escape(__machine_, __first_, __last_));
      return true;
    }

    case '[':
      return __parse_bracket_expression(__machine_, __first_, __last_, __flags_);

    case '(': {
      ++__first_;
      if (__first_ == __last_)
        std::__throw_regex_error<regex_constants::error_paren>();
      auto __next = std::next(__first_);

      bool __explicit_unmarked = (__next != __last_ && *__first_ == '?' && *__next == ':');
      bool __marked            = !(__flags_ & regex_constants::nosubs) && !__explicit_unmarked;
      auto __subexpr_num       = __marked_count_;

      if (__marked) {
        ++__marked_count_;
        __machine_.__push_subexpression_begin(__subexpr_num);
      } else if (__explicit_unmarked) {
        __first_ = ++__next;
      }

      ++__open_count_;
      __parse_exp();
      if (__first_ == __last_ || *__first_ != ')')
        std::__throw_regex_error<regex_constants::error_paren>();
      --__open_count_;
      ++__first_;
      if (__marked)
        __machine_.__push_subexpression_end(__subexpr_num);
      return true;
    }

    case '*':
    case '+':
    case '?':
    case '{':
      std::__throw_regex_error<regex_constants::error_badrepeat>();

    default:
      return __parse_pattern_character();
    }
  }

  bool __parse_term() {
    if (__first_ == __last_)
      return false;

    if (__parse_assertion())
      return true;

    size_t __expr_start = __machine_.size();

    if (!__parse_atom())
      return false;
    __parse_dupl_symbol(__machine_, __first_, __last_, __expr_start, true);
    return true;
  }

  void __parse_alternative() {
    while (__parse_term())
      ;
  }

  void __parse_exp() {
    auto __expr1_start = __machine_.size();
    __parse_alternative();

    while (__first_ != __last_ && *__first_ == '|') {
      ++__first_;
      auto __expr2_start = __machine_.size();
      __parse_alternative();
      __machine_.__push_alternative(__expr1_start, __expr2_start);
    }
  }

public:
  __parser(
      _ForwardIterator __first, _ForwardIterator __last, _Traits __traits, regex_constants::syntax_option_type __flags)
      : __machine_(__traits, false), __first_(__first), __last_(__last), __flags_(__flags) {}

  void __parse_ecma() {
    __parse_exp();
    if (__first_ != __last_)
      std::__throw_regex_error<regex_constants::__re_err_parse>();
    __machine_.__push_end_state();
  }

  __interpreter<_CharT, _Traits> __extract_interpreter() { return std::move(__machine_); }
  size_t mark_count() const { return __marked_count_; }
};
} // namespace __regex::__ecma

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___REGEX_ECMA_PARSER_H
