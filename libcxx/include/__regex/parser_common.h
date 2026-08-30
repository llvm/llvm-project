//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___REGEX_PARSER_COMMON_H
#define _LIBCPP___REGEX_PARSER_COMMON_H

#include <__algorithm/search.h>
#include <__config>
#include <__cstddef/size_t.h>
#include <__iterator/next.h>
#include <__locale_dir/ctype_base.h>
#include <__regex/interpreter.h>
#include <__regex/regex_error.h>
#include <__utility/move.h>
#include <limits>
#include <string>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __regex {

template <class _CharT, class _ForwardIterator>
_CharT __parse_awk_escape(_ForwardIterator& __first, _ForwardIterator __last) {
  switch (*__first) {
  case '\\':
  case '"':
  case '/':
    return *__first++;

  case 'a':
    ++__first;
    return '\a';

  case 'b':
    ++__first;
    return '\b';

  case 'f':
    ++__first;
    return '\f';

  case 'n':
    ++__first;
    return '\n';

  case 'r':
    ++__first;
    return '\r';

  case 't':
    ++__first;
    return '\t';

  case 'v':
    ++__first;
    return '\v';
  }

  auto __is_base_8_char = [](_CharT __c) { return __c >= '0' && __c <= '7'; };

  if (!__is_base_8_char(*__first))
    std::__throw_regex_error<regex_constants::error_escape>();
  unsigned __val = *__first - '0';
  if (++__first != __last && __is_base_8_char(*__first)) {
    __val = 8 * __val + *__first - '0';
    if (++__first != __last && __is_base_8_char(*__first))
      __val = 8 * __val + *__first++ - '0';
  }
  return _CharT(__val);
}

template <class _CharT, class _Traits, class _ForwardIterator>
_CharT __parse_character_escape(
    __interpreter<_CharT, _Traits>& __machine, _ForwardIterator& __first, _ForwardIterator __last) {
  auto __get_next_converted = [&](bool __advance = true) {
    if (__first == __last)
      std::__throw_regex_error<regex_constants::error_escape>();
    auto __converted = __machine.__get_traits().value(*__first, 16);
    if (__advance)
      ++__first;
    if (__converted == -1)
      std::__throw_regex_error<regex_constants::error_escape>();
    return __converted;
  };

  size_t __sum = 0;
  switch (*__first) {
  case 'f':
    ++__first;
    return '\f';
  case 'n':
    ++__first;
    return '\n';
  case 'r':
    ++__first;
    return '\r';
  case 't':
    ++__first;
    return '\t';
  case 'v':
    ++__first;
    return '\v';
  case '0':
    ++__first;
    return '\0';

  case 'c': {
    auto __next = std::next(__first);
    if (__next == __last || !((*__next >= 'A' && *__next <= 'Z') || (*__next >= 'a' && *__next <= 'z')))
      std::__throw_regex_error<regex_constants::error_escape>();
    __first = std::next(__next);
    return _CharT(*__next & ~0x20);
  }

  case 'u': {
    ++__first;
    __sum = __get_next_converted();
    __sum = __sum * 16 + __get_next_converted(false);
  }
    [[__fallthrough__]];
  case 'x': {
    ++__first;
    __sum = __sum * 16 + __get_next_converted();
    __sum = __sum * 16 + __get_next_converted();
    return __sum;
  }
  default: {
    if (__machine.__get_traits().isctype(*__first, ctype_base::alnum))
      std::__throw_regex_error<regex_constants::error_escape>();
    return *__first++;
  }
  }
}

template <class _CharT, class _Traits, class _ForwardIterator>
void __parse_class_escape(
    __interpreter<_CharT, _Traits>& __machine,
    _ForwardIterator& __first,
    _ForwardIterator __last,
    basic_string<_CharT>& __start_range,
    __bracket_expr<_CharT, _Traits>& __buffer) {
  switch (*__first) {
  case '\0':
    ++__first;
    __start_range = '\0';
    return;
  case 'b':
    ++__first;
    __start_range = '\b';
    return;
  case 'd':
    ++__first;
    __buffer.__mask_ |= ctype_base::digit;
    return;
  case 'D':
    ++__first;
    __buffer.__neg_mask_ |= ctype_base::digit;
    return;
  case 's':
    ++__first;
    __buffer.__mask_ |= ctype_base::space;
    return;
  case 'S':
    ++__first;
    __buffer.__neg_mask_ |= ctype_base::space;
    return;
  case 'w':
    ++__first;
    __buffer.__mask_ |= ctype_base::alnum;
    __buffer.__chars_.push_back('_');
    return;
  case 'W':
    ++__first;
    __buffer.__neg_mask_ |= ctype_base::alnum;
    __buffer.__neg_chars_.push_back('_');
    return;
  }
  __start_range = __regex::__parse_character_escape(__machine, __first, __last);
}

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
  } else if (auto __grammar = regex_constants::__get_grammar(__flags);
             __grammar == regex_constants::awk && *__first == '\\') {
    if (++__first == __last)
      return false;
    __start_range = __regex::__parse_awk_escape<_CharT>(__first, __last);
  } else if (__grammar == regex_constants::ECMAScript && *__first == '\\') {
    if (++__first == __last)
      return false;
    __regex::__parse_class_escape<_CharT>(__machine, __first, __last, __start_range, __buffer);
    if (__start_range.empty()) {
      if (__first == __last)
        return false;
      if (*__first == '-')
        std::__throw_regex_error<regex_constants::error_range>();
      return true;
    }
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
    __regex::__parse_collating_symbol(__machine, __end_range, __first, __last);
  } else if (auto __grammar = regex_constants::__get_grammar(__flags);
             __grammar == regex_constants::awk && *__first == '\\') {
    if (++__first == __last)
      return false;
    __start_range = __regex::__parse_awk_escape<_CharT>(__first, __last);
  } else if (__grammar == regex_constants::ECMAScript && *__first == '\\') {
    if (++__first == __last)
      return false;
    __regex::__parse_class_escape(__machine, __first, __last, __end_range, __buffer);
    if (__end_range.empty())
      std::__throw_regex_error<regex_constants::error_range>();
  } else {
    __end_range = *__first;
    ++__first;
  }
  if (__start_range.size() == 1 && __end_range.size() == 1) {
    if (char_traits<_CharT>::lt(__end_range[0], __start_range[0]))
      std::__throw_regex_error<regex_constants::error_range>();
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

// duplication symbol parsing

template <class _CharT, class _Traits, class _ForwardIterator>
bool __parse_dupl_symbol(
    __interpreter<_CharT, _Traits>& __machine,
    _ForwardIterator& __first,
    _ForwardIterator __last,
    size_t __expr_start,
    bool __has_nongreedy = false) {
  if (__first == __last)
    return false;

  auto __check_greediness = [&]() {
    if (!__has_nongreedy)
      return true;
    if (__first == __last || *__first != '?')
      return true;
    ++__first;
    return false;
  };

  switch (*__first) {
  case '*': {
    ++__first;
    __machine.__push_n_to_m_matcher(__expr_start, 0, numeric_limits<size_t>::max(), __check_greediness());
  } break;

  case '+': {
    ++__first;
    __machine.__push_n_to_m_matcher(__expr_start, 1, numeric_limits<size_t>::max(), __check_greediness());
  } break;

  case '?': {
    ++__first;
    __machine.__push_n_to_m_matcher(__expr_start, 0, 1, __check_greediness());
  } break;

  case '{': {
    ++__first;
    size_t __min;
    if (auto __res = __regex::__parse_dup_count(__first, __last)) {
      __min = __res.__num_;
    } else {
      std::__throw_regex_error<regex_constants::error_badbrace>();
    }
    if (__first == __last)
      std::__throw_regex_error<regex_constants::error_brace>();

    switch (*__first) {
    case '}': {
      ++__first;
      __machine.__push_n_to_m_matcher(__expr_start, __min, __min);
      return true;
    } break;

    case ',': {
      ++__first;
      if (__first == __last)
        std::__throw_regex_error<regex_constants::error_badbrace>();
      if (*__first == '}') {
        ++__first;
        __machine.__push_n_to_m_matcher(__expr_start, __min, numeric_limits<size_t>::max());
        return true;
      }

      size_t __max;
      if (auto __res = __regex::__parse_dup_count(__first, __last)) {
        __max = __res.__num_;
      } else {
        std::__throw_regex_error<regex_constants::error_brace>();
      }
      if (__first == __last || *__first != '}')
        std::__throw_regex_error<regex_constants::error_brace>();
      ++__first;
      if (__max < __min)
        std::__throw_regex_error<regex_constants::error_badbrace>();
      __machine.__push_n_to_m_matcher(__expr_start, __min, __max);
      return true;
    } break;
    default:
      std::__throw_regex_error<regex_constants::error_badbrace>();
    }
  } break;
  }
  return false;
}

} // namespace __regex

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___REGEX_BASIC_PARSER_H
