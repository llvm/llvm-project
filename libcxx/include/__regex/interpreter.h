//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___REGEX_INTERPRETER_H
#define _LIBCPP___REGEX_INTERPRETER_H

#include <__algorithm/copy_if.h>
#include <__algorithm/equal.h>
#include <__algorithm/min.h>
#include <__algorithm/reverse.h>
#include <__algorithm/transform.h>
#include <__config>
#include <__iterator/back_insert_iterator.h>
#include <__locale>
#include <__regex/regex_error.h>
#include <__utility/pair.h>
#include <__utility/to_underlying.h>
#include <__vector/vector.h>
#include <stack>
#include <string_view>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _CharT>
class __node;

template <class _BidirectionalIterator>
class sub_match;

template <class _BidirectionalIterator, class _Allocator = allocator<sub_match<_BidirectionalIterator> > >
class match_results;

template <class _CharT>
struct __state {
  enum __states {
    __end_state = -1000,
    __consume_input,          // -999
    __begin_marked_expr,      // -998
    __end_marked_expr,        // -997
    __pop_state,              // -996
    __accept_and_consume,     // -995
    __accept_but_not_consume, // -994
    __reject,                 // -993
    __split,
    __repeat
  };

  int __do_;
  const _CharT* __first_;
  const _CharT* __current_;
  const _CharT* __last_;
  vector<sub_match<const _CharT*> > __sub_matches_;
  vector<pair<size_t, const _CharT*> > __loop_data_;
  const __node<_CharT>* __node_;
  regex_constants::match_flag_type __flags_;
  bool __at_first_;

  _LIBCPP_HIDE_FROM_ABI __state()
      : __do_(0),
        __first_(nullptr),
        __current_(nullptr),
        __last_(nullptr),
        __node_(nullptr),
        __flags_(),
        __at_first_(false) {}
};

namespace __regex {
enum class __state : uint8_t {
  __start_anchor,
  __end_anchor,
  __multiline_start_anchor,
  __multiline_end_anchor,
  __branch_alternative,
  __match_any,
  __match_any_except_newline,
  __match_char,
  __match_icase_char,
  __match_backref,
  __match_icase_backref,
  __match_character_list,
  __match_no_character_list,
  __marked_subexpression_begin,
  __marked_subexpression_end,
  __branch_n_to_m_matcher,
  __branch_nongreedy_n_to_m_matcher,
  __match_n_to_m_times,
  __match_word_boundary,
  __match_no_word_boundary,
  __positive_lookahead,
  __negative_lookahead,
  __relative_jump,
  __end_state,
};

template <class _CharT>
union __interpreter_info {
  __interpreter_info(__state __st) : __state_(__st) {}
  __interpreter_info(_CharT __c) : __char_(__c) {}
  __interpreter_info(uint8_t __int) : __int_(__int) {}

  template <class _Tp>
  __interpreter_info(_Tp) = delete;

  __state __state_;
  _CharT __char_;
  uint8_t __int_;
};

static_assert(sizeof(__interpreter_info<char>) == 1);

template <class _CharT, class _Traits>
struct __bracket_expr {
  vector<_CharT> __chars_;
  vector<_CharT> __neg_chars_;
  vector<_CharT> __digraphs_;
  vector<_CharT> __ranges_;
  vector<typename _Traits::string_type> __equivalences_;
  typename _Traits::char_class_type __mask_     = 0;
  typename _Traits::char_class_type __neg_mask_ = 0;
};

template <class _CharT, class _Traits>
class __interpreter {
  vector<__interpreter_info<_CharT> > __machine_;
  vector<size_t> __initial_loop_values_;
  _Traits __traits_;
  bool __find_longest_;

  static pair<size_t, size_t>
  __read_uleb_impl(_LIBCPP_NOESCAPE const __interpreter_info<_CharT>* __machine, size_t __current_pos) {
    size_t __result = 0;
    size_t __shift  = 0;

    uint8_t __byte;
    do {
      __byte = __machine[__current_pos++].__int_;
      __result |= (__byte & 0x7f) << __shift;
      __shift += 7;
    } while (__byte & 0x80);
    return {__result, __current_pos};
  }

  static size_t __read_uleb(const __interpreter_info<_CharT>* __machine, size_t& __current_pos) {
    auto __val = __machine[__current_pos++].__int_;
    if (__val & 0x80) {
      auto __res    = __read_uleb_impl(__machine, __current_pos);
      __current_pos = __res.second;
      return __res.first;
    }
    return __val;
  }

  struct __global_execution_state {
    struct __local_execution_state;

    vector<__local_execution_state> __states_;
    const __interpreter& __machine_;
    regex_constants::match_flag_type __flags_;

    struct __local_execution_state {
      const _CharT* __current_;
      vector<sub_match<const _CharT*> > __sub_matches_;
      size_t __current_pos_;
      unique_ptr<size_t[]> __loop_values_;
      unique_ptr<const _CharT*[]> __loop_starts_;

      void __copy_to(const _CharT* const __current,
                     vector<__local_execution_state>& __vec,
                     __global_execution_state& __gstate) {
        auto __loop_value_count = __gstate.__machine_.__initial_loop_values_.size();
        auto __loop_values_copy = std::make_unique_for_overwrite<size_t[]>(__loop_value_count);
        std::copy_n(__loop_values_.get(), __loop_value_count, __loop_values_copy.get());
        auto __loop_starts_copy = std::make_unique_for_overwrite<const _CharT*[]>(__loop_value_count);
        std::copy_n(__loop_starts_.get(), __loop_value_count, __loop_starts_copy.get());
        __vec.emplace_back(
            __current, __sub_matches_, __current_pos_, std::move(__loop_values_copy), std::move(__loop_starts_copy));
      }

      static bool __is_word_boundary(const _CharT* __first,
                                     const _CharT* __last,
                                     const _CharT* const __current,
                                     __global_execution_state& __gstate) {
        auto __is_word_char = [&](char __c) {
          return __c == '_' || __gstate.__machine_.__traits_.isctype(__c, ctype_base::alnum);
        };

        if (__first == __last)
          return false;

        if (__current == __last) {
          return !(__gstate.__flags_ & regex_constants::match_not_eow) && __is_word_char(__current[-1]);
        }

        if (__current == __first && !(__gstate.__flags_ & regex_constants::match_prev_avail)) {
          return !(__gstate.__flags_ & regex_constants::match_not_bow) && __is_word_char(__current[0]);
        }

        return __is_word_char(__current[-1]) != __is_word_char(__current[0]);
      }

      struct __charlist_result {
        bool __matched;
        bool __matched_digraph;
        size_t __new_pos;
      };

      static __charlist_result __exec_match_character_list(
          const __interpreter_info<_CharT>* const __code,
          const _CharT* const __last,
          const _CharT* const __current,
          const __global_execution_state& __gstate,
          size_t __current_pos) {
        static constexpr auto __buffer_size =
            std::max(sizeof(typename _Traits::char_class_type) / sizeof(_CharT), size_t(1));
        _CharT __buffer[__buffer_size];
        for (size_t __i = 0; __i != __buffer_size; ++__i)
          __buffer[__i] = __code[__current_pos++].__char_;
        typename _Traits::char_class_type __mask;
        __builtin_memcpy(&__mask, __buffer, sizeof(typename _Traits::char_class_type));

        for (size_t __i = 0; __i != __buffer_size; ++__i)
          __buffer[__i] = __code[__current_pos++].__char_;
        typename _Traits::char_class_type __neg_mask;
        __builtin_memcpy(&__neg_mask, __buffer, sizeof(typename _Traits::char_class_type));

        auto __chars       = __read_uleb(__code, __current_pos);
        auto __digraphs    = __read_uleb(__code, __current_pos) * 2;
        auto __ranges      = __read_uleb(__code, __current_pos) * 4;
        auto __neg_chars   = __read_uleb(__code, __current_pos);
        auto __equiv_count = __read_uleb(__code, __current_pos);
        auto __equiv_size  = __read_uleb(__code, __current_pos);

        bool __found = false;

        for (size_t __i = 0; __i != __chars; ++__i) {
          if (__code[__current_pos + __i].__char_ == *__current) {
            __found = true;
            break;
          }
        }

        __current_pos += __chars;

        if (__found) {
          __current_pos += __digraphs + __ranges + __neg_chars + __equiv_size;
          return {true, false, __current_pos};
        }

        if (__current + 1 != __last) {
          _CharT __vals[2] = {__current[0], __current[1]};

          for (size_t __i = 0; __i != __digraphs; __i += 2) {
            if (__code[__current_pos + __i].__char_ == __vals[0] &&
                __code[__current_pos + __i + 1].__char_ == __vals[1]) {
              __found = true;
              break;
            }
          }
        }

        __current_pos += __digraphs;

        if (__found) {
          __current_pos += __ranges + __neg_chars + __equiv_size;
          return {true, true, __current_pos};
        }

        auto __cmp = __gstate.__machine_.__traits_.transform(__current, __current + 1);
        for (size_t __i = 0; __i != __ranges; __i += 4) {
          _CharT __range_info[4];
          std::transform(__gstate.__machine_.__machine_.begin() + __current_pos + __i,
                         __gstate.__machine_.__machine_.begin() + __current_pos + __i + 4,
                         __range_info,
                         [](__interpreter_info<_CharT> __val) { return __val.__char_; });
          basic_string_view<_CharT> __min(__range_info, __range_info[1] == '\0' ? 1 : 2);
          basic_string_view<_CharT> __max(__range_info + 2, __range_info[3] == '\0' ? 1 : 2);
          if (__min <= __cmp && __cmp <= __max) {
            __found = true;
            break;
          }
        }

        __current_pos += __ranges;

        if (__found) {
          __current_pos += __neg_chars + __equiv_size;
          return {true, false, __current_pos};
        }

        if (__gstate.__machine_.__traits_.isctype(*__current, __mask)) {
          return {true, false, __current_pos};
        }

        bool __none_match = true;
        for (size_t __i = 0; __i != __neg_chars; ++__i) {
          if (__code[__current_pos + __i].__char_ == *__current) {
            __none_match = false;
            break;
          }
        }

        __current_pos += __neg_chars;

        if (__none_match && __neg_mask != 0 && !__gstate.__machine_.__traits_.isctype(*__current, __neg_mask)) {
          return {true, false, __current_pos};
        }

        if (__found) {
          __current_pos += __equiv_size;
          return {true, false, __current_pos};
        }

        auto __after_pos   = __current_pos + __equiv_size;
        auto __transformed = __gstate.__machine_.__traits_.transform_primary(__current, __current + 1);
        for (size_t __i = 0; __i != __equiv_count; ++__i) {
          auto __size = __read_uleb(__code, __current_pos);
          basic_string<_CharT> __str;
          std::transform(__code + __current_pos,
                         __code + __current_pos + __size,
                         std::back_inserter(__str),
                         [](__interpreter_info<_CharT> __v) { return __v.__char_; });
          __current_pos += __size;
          if (__transformed == __str) {
            __found = true;
            break;
          }
        }

        __current_pos = __after_pos;

        return {__found, false, __current_pos};
      }

      pair<bool, size_t> __exec_lookahead(
          const _CharT* const __first,
          const _CharT* const __last,
          const __global_execution_state& __gstate,
          const __interpreter_info<_CharT>* const __code,
          size_t __current_pos,
          bool __is_positive) {
        auto __jump_offset    = __read_uleb(__code, __current_pos);
        auto __submatch_count = __read_uleb(__code, __current_pos);
        auto __submatch_base  = __read_uleb(__code, __current_pos);
        auto __loop_count     = __read_uleb(__code, __current_pos);

        unique_ptr<size_t[]> __initial_loop_values = std::make_unique_for_overwrite<size_t[]>(__loop_count);

        for (size_t __i = 0; __i != __loop_count; ++__i)
          __initial_loop_values[__i] = __read_uleb(__code, __current_pos);

        __global_execution_state __gexec_state{{}, __gstate.__machine_, __gstate.__flags_};
        __gexec_state.__flags_ &= ~regex_constants::__full_match;
        auto& __s = __gexec_state.__states_.emplace_back();
        if (__loop_count > 0)
          __s.__sub_matches_.resize(__submatch_count);
        __s.__current_     = __current_;
        __s.__loop_values_ = std::move(__initial_loop_values);
        __s.__loop_starts_ = std::make_unique<const _CharT*[]>(__loop_count);
        __s.__current_pos_ = __current_pos;
        if (__gexec_state.__execute(__first, __last) != __is_positive)
          return {false, __current_pos};
        if (__is_positive) {
          auto& __matched_state = __gexec_state.__states_.back();
          std::copy(__matched_state.__sub_matches_.begin(),
                    __matched_state.__sub_matches_.end(),
                    __sub_matches_.begin() + __submatch_base);
        }
        __current_pos += __jump_offset;
        return {true, __current_pos};
      }

      size_t __exec_branch_n_to_m_matcher(
          __global_execution_state& __gstate,
          const __interpreter_info<_CharT>* const __code,
          const _CharT* const __current,
          size_t __current_pos,
          bool __is_greedy) {
        auto __loop_index     = __read_uleb(__code, __current_pos);
        auto __again_pos_base = __current_pos;
        auto __again_pos      = __again_pos_base - __read_uleb(__code, __current_pos);
        if (__loop_values_[__loop_index] > 0) {
          --__loop_values_[__loop_index];
          __current_pos = __again_pos;
          return __current_pos;
        }
        if (__loop_starts_[__loop_index] != __current && __loop_values_[__loop_index + 1] > 0) {
          --__loop_values_[__loop_index + 1];
          __loop_starts_[__loop_index] = __current;
          __copy_to(__current, __gstate.__states_, __gstate);
          if (!__is_greedy) {
            __gstate.__states_.back().__current_pos_ = __again_pos;
          } else {
            __gstate.__states_.back().__current_pos_ = __current_pos;
            __current_pos                            = __again_pos;
          }
          return __current_pos;
        }
        return __current_pos;
      }

      pair<bool, size_t> __execute(const _CharT* __first, const _CharT* __last, __global_execution_state& __gstate) {
        const __interpreter_info<_CharT>* const __code = __gstate.__machine().data();
        auto __current_pos                             = __current_pos_;
        auto __counter                                 = 0;
        auto __current                                 = __current_;
        auto __commit_current                          = std::__make_scope_guard([&] { __current_ = __current; });
        while (true) {
          ++__counter;
          auto __st = __code[__current_pos++].__state_;
          switch (__st) {
            using enum __state;

          case __multiline_start_anchor:
            if (!(__gstate.__flags_ & regex_constants::__at_first) && (__current[-1] == '\n' || __current[-1] == '\r'))
              break;
            [[__fallthrough__]];
          case __start_anchor: {
            if (__gstate.__flags_ & regex_constants::match_not_bol ||
                !(__gstate.__flags_ & regex_constants::__at_first) || __first != __current)
              return {false, __counter};
          } break;

          case __multiline_end_anchor:
            if (__current[0] == '\r' || __current[0] == '\n')
              break;
            [[__fallthrough__]];
          case __end_anchor: {
            if (__gstate.__flags_ & regex_constants::match_not_eol || __current != __last)
              return {false, __counter};
          } break;

          case __end_state: {
            return {(!(__gstate.__flags_ & regex_constants::match_not_null) || __current != __first) &&
                        (!(__gstate.__flags_ & regex_constants::match_flag_type::__full_match) || __current == __last),
                    __counter};
          }

          case __match_any: {
            if (__current == __last)
              return {false, __counter};
            ++__current;
          } break;

          case __match_any_except_newline: {
            if (__current == __last || *__current == '\r' || *__current == '\n')
              return {false, __counter};
            if constexpr (__is_same(_CharT, wchar_t)) {
              if (*__current == 0x2028 || *__current == 0x2029)
                return {false, __counter};
            }
            ++__current;
          } break;

          case __match_char: {
            if (__current == __last || *__current != __code[__current_pos++].__char_)
              return {false, __counter};
            ++__current;
          } break;

          case __match_icase_char: {
            if (__current == __last ||
                __gstate.__machine_.__traits_.translate_nocase(*__current) != __code[__current_pos++].__char_)
              return {false, __counter};
            ++__current;
          } break;

          case __branch_alternative: {
            auto __offset = __read_uleb(__code, __current_pos);
            __copy_to(__current, __gstate.__states_, __gstate);
            __gstate.__states_.back().__current_pos_ = __current_pos + __offset;
            break;
          }

          case __relative_jump: {
            __current_pos += __read_uleb(__code, __current_pos);
            break;
          }

          case __branch_n_to_m_matcher:
          case __branch_nongreedy_n_to_m_matcher: {
            __current_pos = __exec_branch_n_to_m_matcher(
                __gstate, __code, __current, __current_pos, __st != __branch_nongreedy_n_to_m_matcher);
          } break;

          case __match_backref: {
            sub_match<const _CharT*>& __match = __sub_matches_[__read_uleb(__code, __current_pos) - 1];
            if (!__match.matched)
              return {false, __counter};

            auto __len = __match.second - __match.first;
            if (__last - __current < __len || !std::equal(__match.first, __match.second, __current))
              return {false, __counter};

            __current += __len;
          } break;

          case __match_icase_backref: {
            sub_match<const _CharT*>& __match = __sub_matches_[__read_uleb(__code, __current_pos) - 1];
            if (!__match.matched)
              return {false, __counter};

            auto __len = __match.second - __match.first;
            if (__last - __current < __len)
              return {false, __counter};

            for (ptrdiff_t __i = 0; __i != __len; ++__i) {
              if (__gstate.__machine_.__traits_.translate_nocase(__match.first[__i]) !=
                  __gstate.__machine_.__traits_.translate_nocase(__current[__i]))
                return {false, __counter};
            }

            __current += __len;
          } break;

          case __match_character_list:
          case __match_no_character_list: {
            if (__current == __last)
              return {false, __counter};
            if (auto [__success, __digraph, __new_pos] =
                    __exec_match_character_list(__code, __last, __current, __gstate, __current_pos);
                __success != (__st == __match_character_list))
              return {false, __counter};
            else {
              __current_pos = __new_pos;
              ++__current; // Swallow the matched character
              if (__digraph)
                ++__current; // Swallow the second character of a digraph
            }
          } break;

          case __marked_subexpression_begin: {
            auto __match                  = __read_uleb(__code, __current_pos);
            __sub_matches_[__match].first = __current;
          } break;

          case __marked_subexpression_end: {
            auto __match                    = __read_uleb(__code, __current_pos);
            __sub_matches_[__match].second  = __current;
            __sub_matches_[__match].matched = true;
          } break;

          case __match_word_boundary: {
            if (!__is_word_boundary(__first, __last, __current, __gstate))
              return {false, __counter};
          } break;

          case __match_no_word_boundary: {
            if (__is_word_boundary(__first, __last, __current, __gstate))
              return {false, __counter};
          } break;

          case __negative_lookahead:
          case __positive_lookahead: {
            if (auto __result =
                    __exec_lookahead(__first, __last, __gstate, __code, __current_pos, (__st == __positive_lookahead));
                __result.first)
              __current_pos = __result.second;
            else
              return {false, __counter};
          } break;

          default:
            std::__libcpp_unreachable();
          }
        }
      }
    };

    const vector<__interpreter_info<_CharT>>& __machine() { return __machine_.__machine_; }

    bool __execute(const _CharT* __first, const _CharT* __last) {
      size_t __length = __last - __first + 1;
      if (__machine_.__find_longest_) {
        __local_execution_state __best_state;
        bool __found_match = false;
        size_t __gcounter  = 0;
        while (!__states_.empty()) {
          if (__gcounter / _LIBCPP_REGEX_COMPLEXITY_FACTOR >= __length)
            std::__throw_regex_error<regex_constants::error_complexity>();
          auto __state = std::move(__states_.back());
          __states_.pop_back();
          auto [__success, __counter] = __state.__execute(__first, __last, *this);
          __gcounter += __counter;
          if (__success) {
            if (!__found_match || __best_state.__current_ < __state.__current_)
              __best_state = std::move(__state);
            if (__best_state.__current_ == __last) {
              __states_.push_back(std::move(__best_state));
              return true;
            }
            __found_match = true;
          }
        }
        __states_.push_back(std::move(__best_state));
        return __found_match;
      } else {
        size_t __gcounter = 0;
        while (!__states_.empty()) {
          if (__gcounter / _LIBCPP_REGEX_COMPLEXITY_FACTOR >= __length)
            std::__throw_regex_error<regex_constants::error_complexity>();
          auto __state = std::move(__states_.back());
          __states_.pop_back();
          if (auto [__success, __counter] = __state.__execute(__first, __last, *this); __success) {
            __states_.push_back(std::move(__state));
            return true;
          } else {
            __gcounter += __counter;
          }
        }
        return false;
      }
    }
  };

  void push_back(__interpreter_info<_CharT> __info) { __machine_.push_back(__info); }

  template <class _Range>
  void append_range(_Range&& __range) {
    __machine_.insert(__machine_.end(), begin(__range), end(__range));
  }

  void insert(size_t __offset, vector<__interpreter_info<_CharT> >& __info) {
    __machine_.insert(__machine_.begin() + __offset, __info.begin(), __info.end());
  }

  template <class _Container>
  static void __write_uleb(_Container& __machine, size_t __val) {
    do {
      uint8_t __byte = __val & 0x7f;
      __val >>= 7;
      if (__val)
        __byte |= 0x80;
      __machine.push_back(__byte);
    } while (__val);
  }

  template <class _Container>
  void __write_string(_Container& __machine, basic_string_view<_CharT> __val) {
    __write_uleb(__machine, __val.size());
    __machine.insert(__machine.end(), __val.begin(), __val.end());
  }

  template <class _Container>
  static void __push_relative_jump(_Container& __machine, size_t __offset) {
    __machine.push_back(__state::__relative_jump);
    __write_uleb(__machine, __offset);
  }

public:
  __interpreter(const _Traits& __traits, bool __find_longest) : __traits_(__traits), __find_longest_(__find_longest) {
    __machine_.reserve(16);
  }

  void __push_any_matcher() { push_back(__state::__match_any); }
  void __push_any_except_newline_matcher() { push_back(__state::__match_any_except_newline); }
  void __push_end_state() { push_back(__state::__end_state); }
  void __push_multiline_start_anchor() { push_back(__state::__multiline_start_anchor); }
  void __push_start_anchor() { push_back(__state::__start_anchor); }
  void __push_multiline_end_anchor() { push_back(__state::__multiline_end_anchor); }
  void __push_end_anchor() { push_back(__state::__end_anchor); }

  void __push_char_matcher(_CharT __char,
                           regex_constants::syntax_option_type __flags = regex_constants::syntax_option_type()) {
    if (__flags & regex_constants::icase) {
      push_back(__state::__match_icase_char);
      push_back(__traits_.translate_nocase(__char));
    } else {
      push_back(__state::__match_char);
      push_back(__char);
    }
  }

  void __push_subexpression_begin(size_t __num) {
    push_back(__state::__marked_subexpression_begin);
    __write_uleb(*this, __num);
  }

  void __push_subexpression_end(size_t __num) {
    push_back(__state::__marked_subexpression_end);
    __write_uleb(*this, __num);
  }

  void __push_n_to_m_matcher(size_t __expr_start, size_t __min, size_t __max, bool __greedy = true) {
    vector<__interpreter_info<_CharT>> __buffer;
    __push_relative_jump(__buffer, size() - __expr_start);
    insert(__expr_start, __buffer);
    __expr_start += __buffer.size();
    push_back(__greedy ? __state::__branch_n_to_m_matcher : __state::__branch_nongreedy_n_to_m_matcher);
    __write_uleb(__machine_, __initial_loop_values_.size());
    __write_uleb(__machine_, size() - __expr_start);
    __initial_loop_values_.push_back(__min);
    __initial_loop_values_.push_back(__max - __min);
  }

  void __push_backref_matcher(size_t __ref,
                              regex_constants::syntax_option_type __flags = regex_constants::syntax_option_type()) {
    push_back((__flags & regex_constants::icase) ? __state::__match_icase_backref : __state::__match_backref);
    __write_uleb(*this, __ref);
  }

  void __push_alternative(size_t __expr1_start, size_t __expr2_start) {
    vector<__interpreter_info<_CharT> > __buffer;
    __push_relative_jump(__buffer, size() - __expr2_start);
    insert(__expr2_start, __buffer);
    __expr2_start += __buffer.size();
    __buffer.clear();
    __buffer.push_back(__state::__branch_alternative);
    __write_uleb(__buffer, __expr2_start - __expr1_start);
    insert(__expr1_start, __buffer);
    __expr2_start += __buffer.size();
  }

  void __push_bracket_expr(bool __negate, const __bracket_expr<_CharT, _Traits>& __expr) {
    push_back(__negate ? __state::__match_no_character_list : __state::__match_character_list);

    _CharT __buffer[std::max(sizeof(typename _Traits::char_class_type) / sizeof(_CharT), size_t(1))];
    __builtin_memcpy(__buffer, &__expr.__mask_, sizeof(__buffer));
    append_range(__buffer);
    __builtin_memcpy(__buffer, &__expr.__neg_mask_, sizeof(__buffer));
    append_range(__buffer);

    basic_string<_CharT> __equivalences_buffer;
    for (auto& __eq : __expr.__equivalences_)
      __write_string(__equivalences_buffer, __eq);
    __write_uleb(*this, __expr.__chars_.size());
    __write_uleb(*this, __expr.__digraphs_.size() / 2);
    __write_uleb(*this, __expr.__ranges_.size() / 4);
    __write_uleb(*this, __expr.__neg_chars_.size());
    __write_uleb(*this, __expr.__equivalences_.size());
    __write_uleb(*this, __equivalences_buffer.size());
    append_range(__expr.__chars_);
    append_range(__expr.__digraphs_);
    append_range(__expr.__ranges_);
    append_range(__expr.__neg_chars_);
    append_range(__equivalences_buffer);
  }

  void __push_word_boundary() { push_back(__state::__match_word_boundary); }
  void __push_no_word_boundary() { push_back(__state::__match_no_word_boundary); }

  struct __lookahead_info {
    vector<size_t> __initial_loop_values_;
    size_t __lookahead_start_;
  };

  __lookahead_info __start_lookahead() {
    __lookahead_info __ret;
    __ret.__initial_loop_values_ = std::move(__initial_loop_values_);
    __ret.__lookahead_start_     = __machine_.size();
    return __ret;
  }

  void __push_lookahead(bool __is_positive, __lookahead_info __info, size_t __marked_base, size_t __marked_count) {
    __push_end_state();
    vector<__interpreter_info<_CharT>> __buffer;
    __buffer.push_back(__is_positive ? __state::__positive_lookahead : __state::__negative_lookahead);
    __write_uleb(__buffer, __machine_.size() - __info.__lookahead_start_);
    __write_uleb(__buffer, __marked_count);
    __write_uleb(__buffer, __marked_base);
    __write_uleb(__buffer, __initial_loop_values_.size());
    for (size_t __val : __initial_loop_values_)
      __write_uleb(__buffer, __val);
    insert(__info.__lookahead_start_, __buffer);
  }

  size_t size() const { return __machine_.size(); }

  pair<bool, const _CharT*>
  __execute(regex_constants::match_flag_type __flags,
            const _CharT* __first,
            const _CharT* __last,
            vector<sub_match<const _CharT*>>& __sub_matches) const {
    using __local_state = typename __global_execution_state::__local_execution_state;
    __local_state __base_state;
    __base_state.__sub_matches_ = std::move(__sub_matches);
    __base_state.__current_     = __first;
    if (!__initial_loop_values_.empty()) {
      __base_state.__loop_values_ = std::make_unique_for_overwrite<size_t[]>(__initial_loop_values_.size());
      std::copy(__initial_loop_values_.begin(), __initial_loop_values_.end(), __base_state.__loop_values_.get());
      __base_state.__loop_starts_ = std::make_unique<const _CharT*[]>(__initial_loop_values_.size());
    }
    __base_state.__current_pos_ = 0;

    __global_execution_state __gexec_state{{}, *this, __flags};
    auto [__base_state_matched, __gcounter] = __base_state.__execute(__first, __last, __gexec_state);

    size_t __length = __last - __first + 1;
    if (__gexec_state.__machine_.__find_longest_) {
      __local_state __best_state = __base_state_matched ? std::move(__base_state) : __local_state{};
      bool __found_match         = __base_state_matched;
      while (!__gexec_state.__states_.empty()) {
        if (__gcounter / _LIBCPP_REGEX_COMPLEXITY_FACTOR >= __length)
          std::__throw_regex_error<regex_constants::error_complexity>();
        auto __state = std::move(__gexec_state.__states_.back());
        __gexec_state.__states_.pop_back();
        auto [__success, __counter] = __state.__execute(__first, __last, __gexec_state);
        __gcounter += __counter;
        if (__success) {
          if (!__found_match || __best_state.__current_ < __state.__current_)
            __best_state = std::move(__state);
          if (__best_state.__current_ == __last) {
            __sub_matches = std::move(__best_state.__sub_matches_);
            return {true, __best_state.__current_};
          }
          __found_match = true;
        }
      }
      if (!__found_match)
        return {false, {}};
      __sub_matches = std::move(__best_state.__sub_matches_);
      return {true, __best_state.__current_};
    } else {
      if (__base_state_matched) {
        __sub_matches = std::move(__base_state.__sub_matches_);
        return {true, __base_state.__current_};
      }
      while (!__gexec_state.__states_.empty()) {
        if (__gcounter / _LIBCPP_REGEX_COMPLEXITY_FACTOR >= __length)
          std::__throw_regex_error<regex_constants::error_complexity>();
        auto __state = std::move(__gexec_state.__states_.back());
        __gexec_state.__states_.pop_back();
        if (auto [__success, __counter] = __state.__execute(__first, __last, __gexec_state); __success) {
          __sub_matches = std::move(__state.__sub_matches_);
          return {true, __state.__current_};
        } else {
          __gcounter += __counter;
        }
      }
      return {false, {}};
    }
  }

  const _Traits& __get_traits() const { return __traits_; }
};
} // namespace __regex

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___REGEX_INTERPRETER_H
