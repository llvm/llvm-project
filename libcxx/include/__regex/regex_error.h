//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___REGEX_REGEX_ERROR_H
#define _LIBCPP___REGEX_REGEX_ERROR_H

#include <__config>
#include <stdexcept>

_LIBCPP_BEGIN_NAMESPACE_STD

namespace regex_constants {

// syntax_option_type

enum syntax_option_type {
  icase    = 1 << 0,
  nosubs   = 1 << 1,
  optimize = 1 << 2,
  collate  = 1 << 3,
#ifdef _LIBCPP_ABI_REGEX_CONSTANTS_NONZERO
  ECMAScript = 1 << 9,
#else
  ECMAScript = 0,
#endif
  basic    = 1 << 4,
  extended = 1 << 5,
  awk      = 1 << 6,
  grep     = 1 << 7,
  egrep    = 1 << 8,
  // 1 << 9 may be used by ECMAScript
  multiline = 1 << 10
};

_LIBCPP_HIDE_FROM_ABI inline _LIBCPP_CONSTEXPR syntax_option_type __get_grammar(syntax_option_type __g) {
#ifdef _LIBCPP_ABI_REGEX_CONSTANTS_NONZERO
  return static_cast<syntax_option_type>(__g & 0x3F0);
#else
  return static_cast<syntax_option_type>(__g & 0x1F0);
#endif
}

inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR syntax_option_type operator~(syntax_option_type __x) {
  return syntax_option_type(~int(__x) & 0x1FF);
}

inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR syntax_option_type
operator&(syntax_option_type __x, syntax_option_type __y) {
  return syntax_option_type(int(__x) & int(__y));
}

inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR syntax_option_type
operator|(syntax_option_type __x, syntax_option_type __y) {
  return syntax_option_type(int(__x) | int(__y));
}

inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR syntax_option_type
operator^(syntax_option_type __x, syntax_option_type __y) {
  return syntax_option_type(int(__x) ^ int(__y));
}

inline _LIBCPP_HIDE_FROM_ABI syntax_option_type& operator&=(syntax_option_type& __x, syntax_option_type __y) {
  __x = __x & __y;
  return __x;
}

inline _LIBCPP_HIDE_FROM_ABI syntax_option_type& operator|=(syntax_option_type& __x, syntax_option_type __y) {
  __x = __x | __y;
  return __x;
}

inline _LIBCPP_HIDE_FROM_ABI syntax_option_type& operator^=(syntax_option_type& __x, syntax_option_type __y) {
  __x = __x ^ __y;
  return __x;
}

// match_flag_type

enum match_flag_type {
  match_default     = 0,
  match_not_bol     = 1 << 0,
  match_not_eol     = 1 << 1,
  match_not_bow     = 1 << 2,
  match_not_eow     = 1 << 3,
  match_any         = 1 << 4,
  match_not_null    = 1 << 5,
  match_continuous  = 1 << 6,
  match_prev_avail  = 1 << 7,
  format_default    = 0,
  format_sed        = 1 << 8,
  format_no_copy    = 1 << 9,
  format_first_only = 1 << 10,
  __no_update_pos   = 1 << 11,
  __full_match      = 1 << 12
};

inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR match_flag_type operator~(match_flag_type __x) {
  return match_flag_type(~int(__x) & 0x0FFF);
}

inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR match_flag_type operator&(match_flag_type __x, match_flag_type __y) {
  return match_flag_type(int(__x) & int(__y));
}

inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR match_flag_type operator|(match_flag_type __x, match_flag_type __y) {
  return match_flag_type(int(__x) | int(__y));
}

inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR match_flag_type operator^(match_flag_type __x, match_flag_type __y) {
  return match_flag_type(int(__x) ^ int(__y));
}

inline _LIBCPP_HIDE_FROM_ABI match_flag_type& operator&=(match_flag_type& __x, match_flag_type __y) {
  __x = __x & __y;
  return __x;
}

inline _LIBCPP_HIDE_FROM_ABI match_flag_type& operator|=(match_flag_type& __x, match_flag_type __y) {
  __x = __x | __y;
  return __x;
}

inline _LIBCPP_HIDE_FROM_ABI match_flag_type& operator^=(match_flag_type& __x, match_flag_type __y) {
  __x = __x ^ __y;
  return __x;
}

enum error_type {
  error_collate = 1,
  error_ctype,
  error_escape,
  error_backref,
  error_brack,
  error_paren,
  error_brace,
  error_badbrace,
  error_range,
  error_space,
  error_badrepeat,
  error_complexity,
  error_stack,
  __re_err_grammar,
  __re_err_empty,
  __re_err_unknown,
  __re_err_parse
};

} // namespace regex_constants

class _LIBCPP_EXPORTED_FROM_ABI regex_error : public runtime_error {
  regex_constants::error_type __code_;

public:
  explicit regex_error(regex_constants::error_type __ecode);
  _LIBCPP_HIDE_FROM_ABI regex_error(const regex_error&) _NOEXCEPT = default;
  ~regex_error() _NOEXCEPT override;
  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI regex_constants::error_type code() const { return __code_; }
};

template <regex_constants::error_type _Ev>
[[__noreturn__]] inline _LIBCPP_HIDE_FROM_ABI void __throw_regex_error() {
#if _LIBCPP_HAS_EXCEPTIONS
  throw regex_error(_Ev);
#else
  _LIBCPP_VERBOSE_ABORT("regex_error was thrown in -fno-exceptions mode");
#endif
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___REGEX_REGEX_ERROR_H
