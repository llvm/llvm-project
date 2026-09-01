//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___LOCALE_DIR_CTYPE_H
#define _LIBCPP___LOCALE_DIR_CTYPE_H

#include <__config>

#if _LIBCPP_HAS_LOCALIZATION

#  include <__cstddef/size_t.h>
#  include <__locale_dir/ctype_base.h>
#  include <__locale_dir/locale.h>
#  include <__locale_dir/locale_base_api.h>
#  include <cctype>
#  include <string>

#  if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#    pragma GCC system_header
#  endif

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

// template <class charT> class ctype

template <class _CharT>
class ctype;

#  if _LIBCPP_HAS_WIDE_CHARACTERS
template <>
class _LIBCPP_EXPORTED_FROM_ABI ctype<wchar_t> : public locale::facet, public ctype_base {
public:
  typedef wchar_t char_type;

  _LIBCPP_HIDE_FROM_ABI explicit ctype(size_t __refs = 0) : locale::facet(__refs) {}

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI bool is(mask __m, char_type __c) const { return do_is(__m, __c); }

  _LIBCPP_HIDE_FROM_ABI const char_type* is(const char_type* __low, const char_type* __high, mask* __vec) const {
    return do_is(__low, __high, __vec);
  }

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI const char_type*
  scan_is(mask __m, const char_type* __low, const char_type* __high) const {
    return do_scan_is(__m, __low, __high);
  }

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI const char_type*
  scan_not(mask __m, const char_type* __low, const char_type* __high) const {
    return do_scan_not(__m, __low, __high);
  }

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI char_type toupper(char_type __c) const { return do_toupper(__c); }

  _LIBCPP_HIDE_FROM_ABI const char_type* toupper(char_type* __low, const char_type* __high) const {
    return do_toupper(__low, __high);
  }

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI char_type tolower(char_type __c) const { return do_tolower(__c); }

  _LIBCPP_HIDE_FROM_ABI const char_type* tolower(char_type* __low, const char_type* __high) const {
    return do_tolower(__low, __high);
  }

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI char_type widen(char __c) const { return do_widen(__c); }

  _LIBCPP_HIDE_FROM_ABI const char* widen(const char* __low, const char* __high, char_type* __to) const {
    return do_widen(__low, __high, __to);
  }

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI char narrow(char_type __c, char __dfault) const {
    return do_narrow(__c, __dfault);
  }

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
#  endif // _LIBCPP_HAS_WIDE_CHARACTERS

inline _LIBCPP_HIDE_FROM_ABI bool __libcpp_isascii(int __c) { return (__c & ~0x7F) == 0; }

template <>
class _LIBCPP_EXPORTED_FROM_ABI ctype<char> : public locale::facet, public ctype_base {
  const mask* __tab_;
  bool __del_;

public:
  typedef char char_type;

  explicit ctype(const mask* __tab = nullptr, bool __del = false, size_t __refs = 0);

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI bool is(mask __m, char_type __c) const {
    return std::__libcpp_isascii(__c) ? (__tab_[static_cast<int>(__c)] & __m) != 0 : false;
  }

  _LIBCPP_HIDE_FROM_ABI const char_type* is(const char_type* __low, const char_type* __high, mask* __vec) const {
    for (; __low != __high; ++__low, ++__vec)
      *__vec = std::__libcpp_isascii(*__low) ? __tab_[static_cast<int>(*__low)] : 0;
    return __low;
  }

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI const char_type*
  scan_is(mask __m, const char_type* __low, const char_type* __high) const {
    for (; __low != __high; ++__low)
      if (std::__libcpp_isascii(*__low) && (__tab_[static_cast<int>(*__low)] & __m))
        break;
    return __low;
  }

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI const char_type*
  scan_not(mask __m, const char_type* __low, const char_type* __high) const {
    for (; __low != __high; ++__low)
      if (!std::__libcpp_isascii(*__low) || !(__tab_[static_cast<int>(*__low)] & __m))
        break;
    return __low;
  }

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI char_type toupper(char_type __c) const { return do_toupper(__c); }

  _LIBCPP_HIDE_FROM_ABI const char_type* toupper(char_type* __low, const char_type* __high) const {
    return do_toupper(__low, __high);
  }

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI char_type tolower(char_type __c) const { return do_tolower(__c); }

  _LIBCPP_HIDE_FROM_ABI const char_type* tolower(char_type* __low, const char_type* __high) const {
    return do_tolower(__low, __high);
  }

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI char_type widen(char __c) const { return do_widen(__c); }

  _LIBCPP_HIDE_FROM_ABI const char* widen(const char* __low, const char* __high, char_type* __to) const {
    return do_widen(__low, __high, __to);
  }

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI char narrow(char_type __c, char __dfault) const {
    return do_narrow(__c, __dfault);
  }

  _LIBCPP_HIDE_FROM_ABI const char*
  narrow(const char_type* __low, const char_type* __high, char __dfault, char* __to) const {
    return do_narrow(__low, __high, __dfault, __to);
  }

  static locale::id id;

#  ifdef _CACHED_RUNES
  static const size_t table_size = _CACHED_RUNES;
#  else
  static const size_t table_size = 256;
#  endif
  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI const mask* table() const _NOEXCEPT { return __tab_; }
  [[__nodiscard__]] static const mask* classic_table() _NOEXCEPT;

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

// template <class CharT> class ctype_byname;

template <class _CharT>
class ctype_byname;

template <>
class _LIBCPP_EXPORTED_FROM_ABI ctype_byname<char> : public ctype<char> {
  __locale::__locale_t __l_;

public:
  explicit ctype_byname(const char*, size_t = 0);
  explicit ctype_byname(const string&, size_t = 0);

protected:
  ~ctype_byname() override;
  char_type do_toupper(char_type) const override;
  const char_type* do_toupper(char_type* __low, const char_type* __high) const override;
  char_type do_tolower(char_type) const override;
  const char_type* do_tolower(char_type* __low, const char_type* __high) const override;
};

#  if _LIBCPP_HAS_WIDE_CHARACTERS
template <>
class _LIBCPP_EXPORTED_FROM_ABI ctype_byname<wchar_t> : public ctype<wchar_t> {
  __locale::__locale_t __l_;

public:
  explicit ctype_byname(const char*, size_t = 0);
  explicit ctype_byname(const string&, size_t = 0);

protected:
  ~ctype_byname() override;
  bool do_is(mask __m, char_type __c) const override;
  const char_type* do_is(const char_type* __low, const char_type* __high, mask* __vec) const override;
  const char_type* do_scan_is(mask __m, const char_type* __low, const char_type* __high) const override;
  const char_type* do_scan_not(mask __m, const char_type* __low, const char_type* __high) const override;
  char_type do_toupper(char_type) const override;
  const char_type* do_toupper(char_type* __low, const char_type* __high) const override;
  char_type do_tolower(char_type) const override;
  const char_type* do_tolower(char_type* __low, const char_type* __high) const override;
  char_type do_widen(char) const override;
  const char* do_widen(const char* __low, const char* __high, char_type* __dest) const override;
  char do_narrow(char_type, char __dfault) const override;
  const char_type*
  do_narrow(const char_type* __low, const char_type* __high, char __dfault, char* __dest) const override;
};
#  endif // _LIBCPP_HAS_WIDE_CHARACTERS

template <class _CharT>
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI bool isspace(_CharT __c, const locale& __loc) {
  return std::use_facet<ctype<_CharT> >(__loc).is(ctype_base::space, __c);
}

template <class _CharT>
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI bool isprint(_CharT __c, const locale& __loc) {
  return std::use_facet<ctype<_CharT> >(__loc).is(ctype_base::print, __c);
}

template <class _CharT>
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI bool iscntrl(_CharT __c, const locale& __loc) {
  return std::use_facet<ctype<_CharT> >(__loc).is(ctype_base::cntrl, __c);
}

template <class _CharT>
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI bool isupper(_CharT __c, const locale& __loc) {
  return std::use_facet<ctype<_CharT> >(__loc).is(ctype_base::upper, __c);
}

template <class _CharT>
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI bool islower(_CharT __c, const locale& __loc) {
  return std::use_facet<ctype<_CharT> >(__loc).is(ctype_base::lower, __c);
}

template <class _CharT>
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI bool isalpha(_CharT __c, const locale& __loc) {
  return std::use_facet<ctype<_CharT> >(__loc).is(ctype_base::alpha, __c);
}

template <class _CharT>
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI bool isdigit(_CharT __c, const locale& __loc) {
  return std::use_facet<ctype<_CharT> >(__loc).is(ctype_base::digit, __c);
}

template <class _CharT>
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI bool ispunct(_CharT __c, const locale& __loc) {
  return std::use_facet<ctype<_CharT> >(__loc).is(ctype_base::punct, __c);
}

template <class _CharT>
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI bool isxdigit(_CharT __c, const locale& __loc) {
  return std::use_facet<ctype<_CharT> >(__loc).is(ctype_base::xdigit, __c);
}

template <class _CharT>
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI bool isalnum(_CharT __c, const locale& __loc) {
  return std::use_facet<ctype<_CharT> >(__loc).is(ctype_base::alnum, __c);
}

template <class _CharT>
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI bool isgraph(_CharT __c, const locale& __loc) {
  return std::use_facet<ctype<_CharT> >(__loc).is(ctype_base::graph, __c);
}

template <class _CharT>
[[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI bool isblank(_CharT __c, const locale& __loc) {
  return std::use_facet<ctype<_CharT> >(__loc).is(ctype_base::blank, __c);
}

template <class _CharT>
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI _CharT toupper(_CharT __c, const locale& __loc) {
  return std::use_facet<ctype<_CharT> >(__loc).toupper(__c);
}

template <class _CharT>
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI _CharT tolower(_CharT __c, const locale& __loc) {
  return std::use_facet<ctype<_CharT> >(__loc).tolower(__c);
}

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_HAS_LOCALIZATION

#endif // _LIBCPP___LOCALE_DIR_CTYPE_H
