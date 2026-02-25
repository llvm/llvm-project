//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___REGEX_ECMA_PARSER_H
#define _LIBCPP___REGEX_ECMA_PARSER_H

#include <__algorithm/search.h>
#include <__config>
#include <__regex/interpreter.h>
#include <__regex/regex_error.h>

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __regex {

// bracket expression parsing

template <class _CharT, class _Traits, class _ForwardIterator>
bool __parse_character_class(
    __interpreter<_CharT, _Traits>& __machine,
    __bracket_expr<_CharT, _Traits>& __buffer,
    _ForwardIterator& __first,
    _ForwardIterator __last,
    regex_constants::syntax_option_type __flags) {
  char __equal_close[2]   = {':', ']'};
  auto __closing_sequence = std::search(__first, __last, __equal_close, __equal_close + 2);

  if (__closing_sequence == __last)
    std::__throw_regex_error<regex_constants::error_brack>();

  auto __class_type =
      __machine.__get_traits().lookup_classname(__first, __closing_sequence, __flags & regex_constants::icase);

  if (__class_type == 0)
    std::__throw_regex_error<regex_constants::error_ctype>();

  __buffer.__mask_ |= __class_type;
  __first = std::next(__closing_sequence, 2);

  return true;
}

template <class _CharT, class _Traits, class _ForwardIterator>
bool __parse_equivalence_class(__interpreter<_CharT, _Traits>& __machine,
                               __bracket_expr<_CharT, _Traits>& __buffer,
                               _ForwardIterator& __first,
                               _ForwardIterator __last) {
  char __equal_close[2]   = {'=', ']'};
  auto __closing_sequence = std::search(__first, __last, __equal_close, __equal_close + 2);

  if (__closing_sequence == __last)
    std::__throw_regex_error<regex_constants::error_brack>();

  auto __collate_name = __machine.__get_traits().lookup_collatename(__first, __closing_sequence);
  if (__collate_name.empty())
    std::__throw_regex_error<regex_constants::error_collate>();

  auto __equiv_name = __machine.__get_traits().transform_primary(__collate_name.begin(), __collate_name.end());
  if (!__equiv_name.empty()) {
    __buffer.__equivalences_.push_back(std::move(__equiv_name));
    __first = std::next(__closing_sequence, 2);
    return true;
  }

  switch (__collate_name.size()) {
  case 1:
    __buffer.__chars_.push_back(__collate_name[0]);
    break;
  case 2:
    __buffer.__digraphs_.push_back(__collate_name[0]);
    __buffer.__digraphs_.push_back(__collate_name[1]);
    break;
  default:
    std::__throw_regex_error<regex_constants::error_collate>();
  }
  __first = std::next(__closing_sequence, 2);
  return true;
}

template <class _CharT, class _Traits, class _ForwardIterator>
void __parse_collating_symbol(__interpreter<_CharT, _Traits>& __machine,
                              basic_string<_CharT>& __buffer,
                              _ForwardIterator& __first,
                              _ForwardIterator __last) {
  char __equal_close[2]   = {'.', ']'};
  auto __closing_sequence = std::search(__first, __last, __equal_close, __equal_close + 2);

  if (__closing_sequence == __last)
    std::__throw_regex_error<regex_constants::error_brack>();

  __buffer = __machine.__get_traits().lookup_collatename(__first, __closing_sequence);
  switch (__buffer.size()) {
  case 1:
  case 2:
    break;
  default:
    std::__throw_regex_error<regex_constants::error_collate>();
  }

  __first = std::next(__closing_sequence, 2);
}

template <class _CharT, class _Traits, class _ForwardIterator>
bool __parse_expression_term(
    __interpreter<_CharT, _Traits>& __machine,
    __bracket_expr<_CharT, _Traits>& __buffer,
    _ForwardIterator& __first,
    _ForwardIterator __last,
    regex_constants::syntax_option_type __flags) {
  _LIBCPP_ASSERT_INTERNAL(__first != __last, "Expected parseable character");

  if (*__first == ']')
    return false;
  auto __next = std::next(__first);
  if (__next == __last)
    return false;

  if (*__first == '[') {
    if (*__next == '=') {
      __first = ++__next;
      return __regex::__parse_equivalence_class(__machine, __buffer, __first, __last);
    }

    if (*__next == ':') {
      __first = ++__next;
      return __regex::__parse_character_class(__machine, __buffer, __first, __last, __flags);
    }
  }

  basic_string<_CharT> __start_range;
  if (*__first == '[' && *__next == '.') {
    __first = ++__next;
    __regex::__parse_collating_symbol(__machine, __start_range, __first, __last);
  } else {
    __start_range = *__first;
    ++__first;
  }

  if (__first == __last)
    return false;

  if (*__first == ']') {
    if (__start_range.size() == 1) {
      __buffer.__chars_.push_back(__start_range[0]);
    } else {
      _LIBCPP_ASSERT_INTERNAL(__start_range.size() == 2, "Unexpected range");
      __buffer.__digraphs_.insert(__buffer.__digraphs_.end(), __start_range.begin(), __start_range.end());
    }
    return true;
  }

  __next = std::next(__first);
  if (__next == __last)
    return false;

  if (*__first != '-' || *__next == ']') {
    if (__start_range.size() == 1) {
      __buffer.__chars_.push_back(__start_range[0]);
    } else {
      _LIBCPP_ASSERT_INTERNAL(__start_range.size() == 2, "Unexpected range");
      __buffer.__digraphs_.insert(__buffer.__digraphs_.end(), __start_range.begin(), __start_range.end());
    }
    return true;
  }

  basic_string<_CharT> __end_range;
  __first = __next;
  ++__next;
  if (__next == __last)
    return false;
  if (*__first == '[' && *__next == '.') {
    __first = ++__next;
    __parse_collating_symbol(__machine, __end_range, __first, __last);
  } else {
    __end_range = *__first;
    ++__first;
  }
  __buffer.__ranges_.push_back(__start_range[0]);
  __buffer.__ranges_.push_back(__start_range.size() == 2 ? __start_range[1] : '\0');
  __buffer.__ranges_.push_back(__end_range[0]);
  __buffer.__ranges_.push_back(__end_range.size() == 2 ? __end_range[1] : '\0');

  return true;
}

template <class _CharT, class _Traits, class _ForwardIterator>
void __parse_follow_list(
    __interpreter<_CharT, _Traits>& __machine,
    __bracket_expr<_CharT, _Traits>& __buffer,
    _ForwardIterator& __first,
    _ForwardIterator __last,
    regex_constants::syntax_option_type __flags) {
  _LIBCPP_ASSERT_INTERNAL(__first != __last, "Expected parseable character");

  while (__regex::__parse_expression_term(__machine, __buffer, __first, __last, __flags))
    ;
}

template <class _CharT, class _Traits, class _ForwardIterator>
bool __parse_bracket_expression(
    __interpreter<_CharT, _Traits>& __machine,
    _ForwardIterator& __first,
    _ForwardIterator __last,
    regex_constants::syntax_option_type __flags) {
  _LIBCPP_ASSERT_INTERNAL(__first != __last, "Expected parseable character");
  if (*__first != '[')
    return false;
  ++__first;
  if (__first == __last)
    std::__throw_regex_error<regex_constants::error_brack>();
  bool __negate = false;
  if (*__first == '^') {
    ++__first;
    if (__first == __last)
      std::__throw_regex_error<regex_constants::error_brack>();
    __negate = true;
  }
  __bracket_expr<_CharT, _Traits> __buffer;

  if (*__first == ']') {
    __buffer.__chars_.push_back(']');
    ++__first;
    if (__first == __last)
      std::__throw_regex_error<regex_constants::error_brack>();
  }

  __regex::__parse_follow_list(__machine, __buffer, __first, __last, __flags);
  if (__first == __last)
    std::__throw_regex_error<regex_constants::error_brack>();
  if (*__first == '-') {
    __buffer.__chars_.push_back('-');
    ++__first;
  }
  if (__first == __last || *__first != ']')
    std::__throw_regex_error<regex_constants::error_brack>();
  ++__first;
  __machine.__push_bracket_expr(__negate, __buffer);
  return true;
}

// dup count parsing

struct __dup_count_result {
  size_t __num_;
  bool __success_ = false;

  __dup_count_result() = default;
  __dup_count_result(size_t __num, bool __success) : __num_(__num), __success_(__success) {}

  operator bool() const { return __success_; }
};

template <class _ForwardIterator>
__dup_count_result __parse_dup_count(_ForwardIterator& __first, _ForwardIterator __last) {
  if (__first == __last || *__first < '0' || *__first > '9')
    return __dup_count_result();
  size_t __num = 0;
  for (; __first != __last && *__first >= '0' && *__first <= '9'; ++__first) {
    if (__num >= numeric_limits<size_t>::max() / 10)
      std::__throw_regex_error<regex_constants::error_badbrace>();
    __num *= 10;
    __num += *__first - '0';
  }
  return __dup_count_result(__num, true);
}

} // namespace __regex

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___REGEX_BASIC_PARSER_H
