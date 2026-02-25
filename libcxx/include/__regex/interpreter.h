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
#include <__ranges/filter_view.h>
#include <__ranges/views.h>
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
  __branch_alternative,
  __match_any,
  __match_char,
  __match_icase_char,
  __match_backref,
  __match_icase_backref,
  __match_character_list,
  __match_no_character_list,
  __marked_subexpression_begin,
  __marked_subexpression_end,
  __branch_n_to_m_matcher,
  __match_n_to_m_times,
  __match_n_times,
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
  vector<_CharT> __digraphs_;
  vector<_CharT> __ranges_;
  vector<typename _Traits::string_type> __equivalences_;
  typename _Traits::char_class_type __mask_ = 0;
};

template <class _CharT, class _Traits>
class __interpreter {
  vector<__interpreter_info<_CharT> > __machine_;
  vector<size_t> __initial_loop_values_;
  _Traits __traits_;

  static size_t __read_uleb(const vector<__interpreter_info<_CharT> >& __machine, size_t& __current_pos) {
    size_t __result = 0;
    size_t __shift  = 0;

    uint8_t __byte;
    do {
      __byte = __machine[__current_pos++].__int_;
      __result |= (__byte & 0x7f) << __shift;
      __shift += 7;
    } while (__byte & 0x80);
    return __result;
  }

  struct __global_execution_state {
    struct __local_execution_state {
      const _CharT* __current_;
      vector<sub_match<const _CharT*> > __sub_matches_;
      size_t __current_pos_;
      vector<size_t> __loop_values_;
      vector<const _CharT*> __loop_starts_;

      bool __execute(const _CharT* __first, const _CharT* __last, __global_execution_state& __gstate) {
        while (true) {
          auto __st = __gstate.__machine()[__current_pos_++].__state_;
          switch (__st) {
            using enum __state;

          case __start_anchor: {
            if (!__gstate.__at_first_ || __first != __current_)
              return false;
          } break;

          case __end_anchor: {
            if (__current_ != __last)
              return false;
          } break;

          case __end_state: {
            return true;
          }

          case __match_any: {
            if (__current_ == __last)
              return false;
            ++__current_;
          } break;

          case __match_char: {
            if (__current_ == __last || *__current_ != __gstate.__machine()[__current_pos_++].__char_)
              return false;
            ++__current_;
          } break;

          case __match_icase_char: {
            if (__current_ == __last || __gstate.__machine_.__traits_.translate_nocase(*__current_) !=
                                            __gstate.__machine()[__current_pos_++].__char_)
              return false;
            ++__current_;
          } break;

          case __branch_alternative: {
            auto __offset = __read_uleb(__gstate.__machine(), __current_pos_);
            auto __cpy    = *this;
            __cpy.__current_pos_ += __offset;
            __gstate.__states_.push(std::move(__cpy));
            break;
          }

          case __relative_jump: {
            __current_pos_ += __read_uleb(__gstate.__machine(), __current_pos_);
            break;
          }

          case __branch_n_to_m_matcher: {
            auto __loop_index     = __read_uleb(__gstate.__machine(), __current_pos_);
            auto __again_pos_base = __current_pos_;
            auto __again_pos      = __again_pos_base - __read_uleb(__gstate.__machine(), __current_pos_);
            if (__loop_values_[__loop_index] > 0) {
              --__loop_values_[__loop_index];
              __current_pos_ = __again_pos;
              break;
            }
            if (__loop_starts_[__loop_index] != __current_ && __loop_values_[__loop_index + 1] > 0) {
              --__loop_values_[__loop_index + 1];
              __loop_starts_[__loop_index] = __current_;
              __gstate.__states_.push(*this);
              __current_pos_ = __again_pos;
              break;
            }
            break;
          }

          case __match_backref: {
            sub_match<const _CharT*>& __match = __sub_matches_[__read_uleb(__gstate.__machine(), __current_pos_) - 1];
            if (!__match.matched)
              return false;

            auto __len = __match.second - __match.first;
            if (__last - __current_ < __len || !std::equal(__match.first, __match.second, __current_))
              return false;

            __current_ += __len;
          } break;

          case __match_icase_backref: {
            sub_match<const _CharT*>& __match = __sub_matches_[__read_uleb(__gstate.__machine(), __current_pos_) - 1];
            if (!__match.matched)
              return false;

            auto __len = __match.second - __match.first;
            if (__last - __current_ < __len)
              return false;

            for (ptrdiff_t __i = 0; __i != __len; ++__i) {
              if (__gstate.__machine_.__traits_.translate_nocase(__match.first[__i]) !=
                  __gstate.__machine_.__traits_.translate_nocase(__current_[__i]))
                return false;
            }

            __current_ += __len;
          } break;

          case __match_character_list:
          case __match_no_character_list: {
            if (__current_ == __last)
              return false;
            static const auto __buffer_size =
                std::max(sizeof(typename _Traits::char_class_type) / sizeof(_CharT), size_t(1));
            _CharT __buffer[__buffer_size];
            for (size_t __i = 0; __i != __buffer_size; ++__i)
              __buffer[__i] = __gstate.__machine()[__current_pos_++].__char_;
            typename _Traits::char_class_type __mask;
            __builtin_memcpy(&__mask, __buffer, sizeof(typename _Traits::char_class_type));

            auto __chars       = __read_uleb(__gstate.__machine(), __current_pos_);
            auto __digraphs    = __read_uleb(__gstate.__machine(), __current_pos_) * 2;
            auto __ranges      = __read_uleb(__gstate.__machine(), __current_pos_) * 4;
            auto __equiv_count = __read_uleb(__gstate.__machine(), __current_pos_);
            auto __equiv_size  = __read_uleb(__gstate.__machine(), __current_pos_);

            bool __found = false;

            for (size_t __i = 0; __i != __chars; ++__i) {
              if (__gstate.__machine()[__current_pos_ + __i].__char_ == *__current_) {
                __found = true;
                break;
              }
            }

            __current_pos_ += __chars;

            if (__found) {
              __current_pos_ += __digraphs + __ranges + __equiv_size;
              ++__current_;
              if (__st == __match_no_character_list)
                return false;
              break;
            }

            if (__current_ + 1 != __last) {
              _CharT __vals[2] = {__current_[0], __current_[1]};

              for (size_t __i = 0; __i != __digraphs; __i += 2) {
                if (__gstate.__machine()[__current_pos_ + __i].__char_ == __vals[0] &&
                    __gstate.__machine()[__current_pos_ + __i + 1].__char_ == __vals[1]) {
                  __found = true;
                  break;
                }
              }
            }

            __current_pos_ += __digraphs;

            if (__found) {
              __current_pos_ += __ranges + __equiv_size;
              __current_ += 2;
              if (__st == __match_no_character_list)
                return false;
              break;
            }

            auto __cmp = __gstate.__machine_.__traits_.transform(__current_, __current_ + 1);
            for (size_t __i = 0; __i != __ranges; __i += 4) {
              _CharT __range_info[4];
              std::transform(__gstate.__machine_.__machine_.begin() + __current_pos_ + __i,
                             __gstate.__machine_.__machine_.begin() + __current_pos_ + __i + 4,
                             __range_info,
                             [](__interpreter_info<_CharT> __val) { return __val.__char_; });
              basic_string_view<_CharT> __min(__range_info, __range_info[1] == '\0' ? 1 : 2);
              basic_string_view<_CharT> __max(__range_info + 2, __range_info[3] == '\0' ? 1 : 2);
              if (__min <= __cmp && __cmp <= __max) {
                __found = true;
                __current_ += __cmp.size();
                break;
              }
            }

            __current_pos_ += __ranges;

            if (__found) {
              __current_pos_ += __equiv_size;
              if (__st == __match_no_character_list)
                return false;
              break;
            }

            if (__gstate.__machine_.__traits_.isctype(*__current_, __mask)) {
              ++__current_;
              if (__st == __match_no_character_list)
                return false;
              break;
            }

            auto __after_pos   = __current_pos_ + __equiv_size;
            auto __transformed = __gstate.__machine_.__traits_.transform_primary(__current_, __current_ + 1);
            for (size_t __i = 0; __i != __equiv_count; ++__i) {
              auto __size = __read_uleb(__gstate.__machine(), __current_pos_);
              basic_string<_CharT> __str;
              std::transform(__gstate.__machine().begin() + __current_pos_,
                             __gstate.__machine().begin() + __current_pos_ + __size,
                             std::back_inserter(__str),
                             [](__interpreter_info<_CharT> __v) { return __v.__char_; });
              __current_pos_ += __size;
              if (__transformed == __str) {
                __found = true;
                break;
              }
            }

            __current_pos_ = __after_pos;

            if (__found) {
              ++__current_;
              if (__st == __match_no_character_list)
                return false;
              break;
            }

            if (__st == __match_no_character_list) {
              ++__current_;
              break;
            }
            return false;

          } break;

          case __marked_subexpression_begin: {
            auto __match                  = __read_uleb(__gstate.__machine(), __current_pos_);
            __sub_matches_[__match].first = __current_;
          } break;

          case __marked_subexpression_end: {
            auto __match                    = __read_uleb(__gstate.__machine(), __current_pos_);
            __sub_matches_[__match].second  = __current_;
            __sub_matches_[__match].matched = true;
          } break;

          default:
            std::__libcpp_unreachable();
          }
        }
      }
    };

    bool __at_first_;
    stack<__local_execution_state> __states_;
    const __interpreter& __machine_;

    __global_execution_state(bool __at_first, __local_execution_state __initial_state, const __interpreter& __machine)
        : __at_first_(__at_first), __machine_(__machine) {
      __states_.push(std::move(__initial_state));
    }

    const vector<__interpreter_info<_CharT> >& __machine() { return __machine_.__machine_; }

    bool __execute(const _CharT* __first, const _CharT* __last) {
      while (!__states_.empty()) {
        auto __st = std::move(__states_.top());
        __states_.pop();
        if (__st.__execute(__first, __last, *this)) {
          __states_.push(std::move(__st));
          return true;
        }
      }
      return false;
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
  __interpreter(_Traits __traits) : __traits_(__traits) {}

  void __push_any_matcher() { push_back(__state::__match_any); }
  void __push_end_state() { push_back(__state::__end_state); }
  void __push_start_anchor() { push_back(__state::__start_anchor); }
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

  void __push_n_matcher(size_t __expr_start, size_t __count) {
    vector<__interpreter_info<_CharT> > __buffer;
    __buffer.push_back(__state::__match_n_times);
    __write_uleb(__buffer, __count);
    __write_uleb(__buffer, __machine_.size() - __expr_start + 1);
    insert(__expr_start, __buffer);
    push_back(__state::__end_state);
  }

  void __push_n_to_m_matcher(size_t __expr_start, size_t __min, size_t __max) {
    vector<__interpreter_info<_CharT> > __buffer;
    __push_relative_jump(__buffer, size() - __expr_start);
    insert(__expr_start, __buffer);
    __expr_start += __buffer.size();
    push_back(__state::__branch_n_to_m_matcher);
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
    __buffer.push_back(__state::__branch_alternative);
    __write_uleb(__buffer, __expr2_start - __expr1_start);
    insert(__expr1_start, __buffer);
    __expr2_start += __buffer.size();
    __buffer.clear();
    __push_relative_jump(__buffer, size() - __expr2_start);
    insert(__expr2_start, __buffer);
  }

  void __push_bracket_expr(bool __negate, const __bracket_expr<_CharT, _Traits>& __expr) {
    push_back(__negate ? __state::__match_no_character_list : __state::__match_character_list);

    _CharT __buffer[std::max(sizeof(typename _Traits::char_class_type) / sizeof(_CharT), size_t(1))];
    __builtin_memcpy(__buffer, &__expr.__mask_, sizeof(__buffer));
    append_range(__buffer);

    basic_string<_CharT> __equivalences_buffer;
    for (auto& __eq : __expr.__equivalences_)
      __write_string(__equivalences_buffer, __eq);
    __write_uleb(*this, __expr.__chars_.size());
    __write_uleb(*this, __expr.__digraphs_.size() / 2);
    __write_uleb(*this, __expr.__ranges_.size() / 4);
    __write_uleb(*this, __expr.__equivalences_.size());
    __write_uleb(*this, __equivalences_buffer.size());
    append_range(__expr.__chars_);
    append_range(__expr.__digraphs_);
    append_range(__expr.__ranges_);
    append_range(__equivalences_buffer);
  }

  size_t size() const { return __machine_.size(); }

  __global_execution_state
  __get_exec_state(vector<sub_match<const _CharT*> >& __sub_matches, const _CharT* __current, bool __at_first) const {
    typename __global_execution_state::__local_execution_state __base_state;
    __base_state.__sub_matches_ = __sub_matches;
    __base_state.__current_     = __current;
    __base_state.__loop_values_ = __initial_loop_values_;
    __base_state.__loop_starts_.resize(__initial_loop_values_.size());
    __base_state.__current_pos_ = 0;
    return __global_execution_state(__at_first, __base_state, *this);
  }

  _Traits __get_traits() { return __traits_; }
};
} // namespace __regex

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___REGEX_INTERPRETER_H
