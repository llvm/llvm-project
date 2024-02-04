//===-----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___LOCALE_CTYPE_H
#define _LIBCPP___LOCALE_CTYPE_H

#include <__config>
#include <__locale_dir/ctype_base.h>
#include <__locale_dir/locale.h>
#include <__locale_dir/locale_base_api.h>
#include <cstddef>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _CharT>
class _LIBCPP_TEMPLATE_VIS ctype;

#ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
template <>
class _LIBCPP_EXPORTED_FROM_ABI ctype<wchar_t> : public locale::facet, public ctype_base {
public:
  typedef wchar_t char_type;

  _LIBCPP_HIDE_FROM_ABI explicit ctype(size_t __refs = 0) : locale::facet(__refs) {}

  _LIBCPP_HIDE_FROM_ABI bool is(mask __m, char_type __c) const { return do_is(__m, __c); }

  _LIBCPP_HIDE_FROM_ABI const char_type* is(const char_type* __low, const char_type* __high, mask* __vec) const {
    return do_is(__low, __high, __vec);
  }

  _LIBCPP_HIDE_FROM_ABI const char_type* scan_is(mask __m, const char_type* __low, const char_type* __high) const {
    return do_scan_is(__m, __low, __high);
  }

  _LIBCPP_HIDE_FROM_ABI const char_type* scan_not(mask __m, const char_type* __low, const char_type* __high) const {
    return do_scan_not(__m, __low, __high);
  }

  _LIBCPP_HIDE_FROM_ABI char_type toupper(char_type __c) const { return do_toupper(__c); }

  _LIBCPP_HIDE_FROM_ABI const char_type* toupper(char_type* __low, const char_type* __high) const {
    return do_toupper(__low, __high);
  }

  _LIBCPP_HIDE_FROM_ABI char_type tolower(char_type __c) const { return do_tolower(__c); }

  _LIBCPP_HIDE_FROM_ABI const char_type* tolower(char_type* __low, const char_type* __high) const {
    return do_tolower(__low, __high);
  }

  _LIBCPP_HIDE_FROM_ABI char_type widen(char __c) const { return do_widen(__c); }

  _LIBCPP_HIDE_FROM_ABI const char* widen(const char* __low, const char* __high, char_type* __to) const {
    return do_widen(__low, __high, __to);
  }

  _LIBCPP_HIDE_FROM_ABI char narrow(char_type __c, char __dfault) const { return do_narrow(__c, __dfault); }

  _LIBCPP_HIDE_FROM_ABI const char_type*
  narrow(const char_type* __low, const char_type* __high, char __dfault, char* __to) const {
    return do_narrow(__low, __high, __dfault, __to);
  }

  static locale::id id;

protected:
  ~ctype() override;
  virtual bool do_is(mask __m, char_type __c) const;
  virtual const char_type* do_is(const char_type* __low, const char_type* __high, mask* __vec) const;
  virtual const char_type* do_scan_is(mask __m, const char_type* __low, const char_type* __high) const;
  virtual const char_type* do_scan_not(mask __m, const char_type* __low, const char_type* __high) const;
  virtual char_type do_toupper(char_type) const;
  virtual const char_type* do_toupper(char_type* __low, const char_type* __high) const;
  virtual char_type do_tolower(char_type) const;
  virtual const char_type* do_tolower(char_type* __low, const char_type* __high) const;
  virtual char_type do_widen(char) const;
  virtual const char* do_widen(const char* __low, const char* __high, char_type* __dest) const;
  virtual char do_narrow(char_type, char __dfault) const;
  virtual const char_type*
  do_narrow(const char_type* __low, const char_type* __high, char __dfault, char* __dest) const;
};
#endif // _LIBCPP_HAS_NO_WIDE_CHARACTERS

template <>
class _LIBCPP_EXPORTED_FROM_ABI ctype<char> : public locale::facet, public ctype_base {
  const mask* __tab_;
  bool __del_;

public:
  typedef char char_type;

  explicit ctype(const mask* __tab = nullptr, bool __del = false, size_t __refs = 0);

  _LIBCPP_HIDE_FROM_ABI bool is(mask __m, char_type __c) const {
    return isascii(__c) ? (__tab_[static_cast<int>(__c)] & __m) != 0 : false;
  }

  _LIBCPP_HIDE_FROM_ABI const char_type* is(const char_type* __low, const char_type* __high, mask* __vec) const {
    for (; __low != __high; ++__low, ++__vec)
      *__vec = isascii(*__low) ? __tab_[static_cast<int>(*__low)] : 0;
    return __low;
  }

  _LIBCPP_HIDE_FROM_ABI const char_type* scan_is(mask __m, const char_type* __low, const char_type* __high) const {
    for (; __low != __high; ++__low)
      if (isascii(*__low) && (__tab_[static_cast<int>(*__low)] & __m))
        break;
    return __low;
  }

  _LIBCPP_HIDE_FROM_ABI const char_type* scan_not(mask __m, const char_type* __low, const char_type* __high) const {
    for (; __low != __high; ++__low)
      if (!isascii(*__low) || !(__tab_[static_cast<int>(*__low)] & __m))
        break;
    return __low;
  }

  _LIBCPP_HIDE_FROM_ABI char_type toupper(char_type __c) const { return do_toupper(__c); }

  _LIBCPP_HIDE_FROM_ABI const char_type* toupper(char_type* __low, const char_type* __high) const {
    return do_toupper(__low, __high);
  }

  _LIBCPP_HIDE_FROM_ABI char_type tolower(char_type __c) const { return do_tolower(__c); }

  _LIBCPP_HIDE_FROM_ABI const char_type* tolower(char_type* __low, const char_type* __high) const {
    return do_tolower(__low, __high);
  }

  _LIBCPP_HIDE_FROM_ABI char_type widen(char __c) const { return do_widen(__c); }

  _LIBCPP_HIDE_FROM_ABI const char* widen(const char* __low, const char* __high, char_type* __to) const {
    return do_widen(__low, __high, __to);
  }

  _LIBCPP_HIDE_FROM_ABI char narrow(char_type __c, char __dfault) const { return do_narrow(__c, __dfault); }

  _LIBCPP_HIDE_FROM_ABI const char*
  narrow(const char_type* __low, const char_type* __high, char __dfault, char* __to) const {
    return do_narrow(__low, __high, __dfault, __to);
  }

  static locale::id id;

#ifdef _CACHED_RUNES
  static const size_t table_size = _CACHED_RUNES;
#else
  static const size_t table_size = 256; // FIXME: Don't hardcode this.
#endif
  _LIBCPP_HIDE_FROM_ABI const mask* table() const _NOEXCEPT { return __tab_; }
  static const mask* classic_table() _NOEXCEPT;
#if defined(__GLIBC__) || defined(__EMSCRIPTEN__)
  static const int* __classic_upper_table() _NOEXCEPT;
  static const int* __classic_lower_table() _NOEXCEPT;
#endif
#if defined(__NetBSD__)
  static const short* __classic_upper_table() _NOEXCEPT;
  static const short* __classic_lower_table() _NOEXCEPT;
#endif
#if defined(__MVS__)
  static const unsigned short* __classic_upper_table() _NOEXCEPT;
  static const unsigned short* __classic_lower_table() _NOEXCEPT;
#endif

protected:
  ~ctype() override;
  virtual char_type do_toupper(char_type __c) const;
  virtual const char_type* do_toupper(char_type* __low, const char_type* __high) const;
  virtual char_type do_tolower(char_type __c) const;
  virtual const char_type* do_tolower(char_type* __low, const char_type* __high) const;
  virtual char_type do_widen(char __c) const;
  virtual const char* do_widen(const char* __low, const char* __high, char_type* __to) const;
  virtual char do_narrow(char_type __c, char __dfault) const;
  virtual const char* do_narrow(const char_type* __low, const char_type* __high, char __dfault, char* __to) const;
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___LOCALE_CTYPE_H
