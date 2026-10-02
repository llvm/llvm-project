//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___LOCALE_DIR_CODECVT_H
#define _LIBCPP___LOCALE_DIR_CODECVT_H

#include <__config>

#if _LIBCPP_HAS_LOCALIZATION

#  include <__cstddef/size_t.h>
#  include <__locale_dir/locale.h>
#  include <__locale_dir/locale_base_api.h>
#  include <string>

#  if _LIBCPP_HAS_WIDE_CHARACTERS
#    include <cwchar>
#  else
#    include <__std_mbstate_t.h>
#  endif

#  if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#    pragma GCC system_header
#  endif

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

// codecvt_base

class _LIBCPP_EXPORTED_FROM_ABI codecvt_base {
public:
  _LIBCPP_HIDE_FROM_ABI codecvt_base() {}
  enum result { ok, partial, error, noconv };
};

// template <class internT, class externT, class stateT> class codecvt;

template <class _InternT, class _ExternT, class _StateT>
class codecvt;

// template <> class codecvt<char, char, mbstate_t>

template <>
class _LIBCPP_EXPORTED_FROM_ABI codecvt<char, char, mbstate_t> : public locale::facet, public codecvt_base {
public:
  typedef char intern_type;
  typedef char extern_type;
  typedef mbstate_t state_type;

  _LIBCPP_HIDE_FROM_ABI explicit codecvt(size_t __refs = 0) : locale::facet(__refs) {}

  _LIBCPP_HIDE_FROM_ABI result
  out(state_type& __st,
      const intern_type* __frm,
      const intern_type* __frm_end,
      const intern_type*& __frm_nxt,
      extern_type* __to,
      extern_type* __to_end,
      extern_type*& __to_nxt) const {
    return do_out(__st, __frm, __frm_end, __frm_nxt, __to, __to_end, __to_nxt);
  }

  _LIBCPP_HIDE_FROM_ABI result
  unshift(state_type& __st, extern_type* __to, extern_type* __to_end, extern_type*& __to_nxt) const {
    return do_unshift(__st, __to, __to_end, __to_nxt);
  }

  _LIBCPP_HIDE_FROM_ABI result
  in(state_type& __st,
     const extern_type* __frm,
     const extern_type* __frm_end,
     const extern_type*& __frm_nxt,
     intern_type* __to,
     intern_type* __to_end,
     intern_type*& __to_nxt) const {
    return do_in(__st, __frm, __frm_end, __frm_nxt, __to, __to_end, __to_nxt);
  }

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI int encoding() const _NOEXCEPT { return do_encoding(); }

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI bool always_noconv() const _NOEXCEPT { return do_always_noconv(); }

  _LIBCPP_HIDE_FROM_ABI int
  length(state_type& __st, const extern_type* __frm, const extern_type* __end, size_t __mx) const {
    return do_length(__st, __frm, __end, __mx);
  }

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI int max_length() const _NOEXCEPT { return do_max_length(); }

  static locale::id id;

protected:
  _LIBCPP_HIDE_FROM_ABI explicit codecvt(const char*, size_t __refs = 0) : locale::facet(__refs) {}

  ~codecvt() override;

  virtual result
  do_out(state_type& __st,
         const intern_type* __frm,
         const intern_type* __frm_end,
         const intern_type*& __frm_nxt,
         extern_type* __to,
         extern_type* __to_end,
         extern_type*& __to_nxt) const;
  virtual result
  do_in(state_type& __st,
        const extern_type* __frm,
        const extern_type* __frm_end,
        const extern_type*& __frm_nxt,
        intern_type* __to,
        intern_type* __to_end,
        intern_type*& __to_nxt) const;
  virtual result do_unshift(state_type& __st, extern_type* __to, extern_type* __to_end, extern_type*& __to_nxt) const;
  virtual int do_encoding() const _NOEXCEPT;
  virtual bool do_always_noconv() const _NOEXCEPT;
  virtual int do_length(state_type& __st, const extern_type* __frm, const extern_type* __end, size_t __mx) const;
  virtual int do_max_length() const _NOEXCEPT;
};

// template <> class codecvt<wchar_t, char, mbstate_t>

#  if _LIBCPP_HAS_WIDE_CHARACTERS
template <>
class _LIBCPP_EXPORTED_FROM_ABI codecvt<wchar_t, char, mbstate_t> : public locale::facet, public codecvt_base {
  __locale::__locale_t __l_;

public:
  typedef wchar_t intern_type;
  typedef char extern_type;
  typedef mbstate_t state_type;

  explicit codecvt(size_t __refs = 0);

  _LIBCPP_HIDE_FROM_ABI result
  out(state_type& __st,
      const intern_type* __frm,
      const intern_type* __frm_end,
      const intern_type*& __frm_nxt,
      extern_type* __to,
      extern_type* __to_end,
      extern_type*& __to_nxt) const {
    return do_out(__st, __frm, __frm_end, __frm_nxt, __to, __to_end, __to_nxt);
  }

  _LIBCPP_HIDE_FROM_ABI result
  unshift(state_type& __st, extern_type* __to, extern_type* __to_end, extern_type*& __to_nxt) const {
    return do_unshift(__st, __to, __to_end, __to_nxt);
  }

  _LIBCPP_HIDE_FROM_ABI result
  in(state_type& __st,
     const extern_type* __frm,
     const extern_type* __frm_end,
     const extern_type*& __frm_nxt,
     intern_type* __to,
     intern_type* __to_end,
     intern_type*& __to_nxt) const {
    return do_in(__st, __frm, __frm_end, __frm_nxt, __to, __to_end, __to_nxt);
  }

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI int encoding() const _NOEXCEPT { return do_encoding(); }

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI bool always_noconv() const _NOEXCEPT { return do_always_noconv(); }

  _LIBCPP_HIDE_FROM_ABI int
  length(state_type& __st, const extern_type* __frm, const extern_type* __end, size_t __mx) const {
    return do_length(__st, __frm, __end, __mx);
  }

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI int max_length() const _NOEXCEPT { return do_max_length(); }

  static locale::id id;

protected:
  explicit codecvt(const char*, size_t __refs = 0);

  ~codecvt() override;

  virtual result
  do_out(state_type& __st,
         const intern_type* __frm,
         const intern_type* __frm_end,
         const intern_type*& __frm_nxt,
         extern_type* __to,
         extern_type* __to_end,
         extern_type*& __to_nxt) const;
  virtual result
  do_in(state_type& __st,
        const extern_type* __frm,
        const extern_type* __frm_end,
        const extern_type*& __frm_nxt,
        intern_type* __to,
        intern_type* __to_end,
        intern_type*& __to_nxt) const;
  virtual result do_unshift(state_type& __st, extern_type* __to, extern_type* __to_end, extern_type*& __to_nxt) const;
  virtual int do_encoding() const _NOEXCEPT;
  virtual bool do_always_noconv() const _NOEXCEPT;
  virtual int do_length(state_type&, const extern_type* __frm, const extern_type* __end, size_t __mx) const;
  virtual int do_max_length() const _NOEXCEPT;
};
#  endif // _LIBCPP_HAS_WIDE_CHARACTERS

// template <> class codecvt<char16_t, char, mbstate_t> // deprecated in C++20

template <>
class _LIBCPP_DEPRECATED_IN_CXX20 _LIBCPP_EXPORTED_FROM_ABI codecvt<char16_t, char, mbstate_t>
    : public locale::facet, public codecvt_base {
public:
  typedef char16_t intern_type;
  typedef char extern_type;
  typedef mbstate_t state_type;

  _LIBCPP_HIDE_FROM_ABI explicit codecvt(size_t __refs = 0) : locale::facet(__refs) {}

  _LIBCPP_HIDE_FROM_ABI result
  out(state_type& __st,
      const intern_type* __frm,
      const intern_type* __frm_end,
      const intern_type*& __frm_nxt,
      extern_type* __to,
      extern_type* __to_end,
      extern_type*& __to_nxt) const {
    return do_out(__st, __frm, __frm_end, __frm_nxt, __to, __to_end, __to_nxt);
  }

  _LIBCPP_HIDE_FROM_ABI result
  unshift(state_type& __st, extern_type* __to, extern_type* __to_end, extern_type*& __to_nxt) const {
    return do_unshift(__st, __to, __to_end, __to_nxt);
  }

  _LIBCPP_HIDE_FROM_ABI result
  in(state_type& __st,
     const extern_type* __frm,
     const extern_type* __frm_end,
     const extern_type*& __frm_nxt,
     intern_type* __to,
     intern_type* __to_end,
     intern_type*& __to_nxt) const {
    return do_in(__st, __frm, __frm_end, __frm_nxt, __to, __to_end, __to_nxt);
  }

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI int encoding() const _NOEXCEPT { return do_encoding(); }

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI bool always_noconv() const _NOEXCEPT { return do_always_noconv(); }

  _LIBCPP_HIDE_FROM_ABI int
  length(state_type& __st, const extern_type* __frm, const extern_type* __end, size_t __mx) const {
    return do_length(__st, __frm, __end, __mx);
  }

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI int max_length() const _NOEXCEPT { return do_max_length(); }

  static locale::id id;

protected:
  _LIBCPP_HIDE_FROM_ABI explicit codecvt(const char*, size_t __refs = 0) : locale::facet(__refs) {}

  ~codecvt() override;

  virtual result
  do_out(state_type& __st,
         const intern_type* __frm,
         const intern_type* __frm_end,
         const intern_type*& __frm_nxt,
         extern_type* __to,
         extern_type* __to_end,
         extern_type*& __to_nxt) const;
  virtual result
  do_in(state_type& __st,
        const extern_type* __frm,
        const extern_type* __frm_end,
        const extern_type*& __frm_nxt,
        intern_type* __to,
        intern_type* __to_end,
        intern_type*& __to_nxt) const;
  virtual result do_unshift(state_type& __st, extern_type* __to, extern_type* __to_end, extern_type*& __to_nxt) const;
  virtual int do_encoding() const _NOEXCEPT;
  virtual bool do_always_noconv() const _NOEXCEPT;
  virtual int do_length(state_type&, const extern_type* __frm, const extern_type* __end, size_t __mx) const;
  virtual int do_max_length() const _NOEXCEPT;
};

#  if _LIBCPP_HAS_CHAR8_T

// template <> class codecvt<char16_t, char8_t, mbstate_t> // C++20, deprecated in C++20

template <>
class _LIBCPP_DEPRECATED_IN_CXX20 _LIBCPP_EXPORTED_FROM_ABI codecvt<char16_t, char8_t, mbstate_t>
    : public locale::facet, public codecvt_base {
public:
  typedef char16_t intern_type;
  typedef char8_t extern_type;
  typedef mbstate_t state_type;

  _LIBCPP_HIDE_FROM_ABI explicit codecvt(size_t __refs = 0) : locale::facet(__refs) {}

  _LIBCPP_HIDE_FROM_ABI result
  out(state_type& __st,
      const intern_type* __frm,
      const intern_type* __frm_end,
      const intern_type*& __frm_nxt,
      extern_type* __to,
      extern_type* __to_end,
      extern_type*& __to_nxt) const {
    return do_out(__st, __frm, __frm_end, __frm_nxt, __to, __to_end, __to_nxt);
  }

  _LIBCPP_HIDE_FROM_ABI result
  unshift(state_type& __st, extern_type* __to, extern_type* __to_end, extern_type*& __to_nxt) const {
    return do_unshift(__st, __to, __to_end, __to_nxt);
  }

  _LIBCPP_HIDE_FROM_ABI result
  in(state_type& __st,
     const extern_type* __frm,
     const extern_type* __frm_end,
     const extern_type*& __frm_nxt,
     intern_type* __to,
     intern_type* __to_end,
     intern_type*& __to_nxt) const {
    return do_in(__st, __frm, __frm_end, __frm_nxt, __to, __to_end, __to_nxt);
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI int encoding() const _NOEXCEPT { return do_encoding(); }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI bool always_noconv() const _NOEXCEPT { return do_always_noconv(); }

  _LIBCPP_HIDE_FROM_ABI int
  length(state_type& __st, const extern_type* __frm, const extern_type* __end, size_t __mx) const {
    return do_length(__st, __frm, __end, __mx);
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI int max_length() const _NOEXCEPT { return do_max_length(); }

  static locale::id id;

protected:
  _LIBCPP_HIDE_FROM_ABI explicit codecvt(const char*, size_t __refs = 0) : locale::facet(__refs) {}

  ~codecvt() override;

  virtual result
  do_out(state_type& __st,
         const intern_type* __frm,
         const intern_type* __frm_end,
         const intern_type*& __frm_nxt,
         extern_type* __to,
         extern_type* __to_end,
         extern_type*& __to_nxt) const;
  virtual result
  do_in(state_type& __st,
        const extern_type* __frm,
        const extern_type* __frm_end,
        const extern_type*& __frm_nxt,
        intern_type* __to,
        intern_type* __to_end,
        intern_type*& __to_nxt) const;
  virtual result do_unshift(state_type& __st, extern_type* __to, extern_type* __to_end, extern_type*& __to_nxt) const;
  virtual int do_encoding() const _NOEXCEPT;
  virtual bool do_always_noconv() const _NOEXCEPT;
  virtual int do_length(state_type&, const extern_type* __frm, const extern_type* __end, size_t __mx) const;
  virtual int do_max_length() const _NOEXCEPT;
};

#  endif

// template <> class codecvt<char32_t, char, mbstate_t> // deprecated in C++20

template <>
class _LIBCPP_DEPRECATED_IN_CXX20 _LIBCPP_EXPORTED_FROM_ABI codecvt<char32_t, char, mbstate_t>
    : public locale::facet, public codecvt_base {
public:
  typedef char32_t intern_type;
  typedef char extern_type;
  typedef mbstate_t state_type;

  _LIBCPP_HIDE_FROM_ABI explicit codecvt(size_t __refs = 0) : locale::facet(__refs) {}

  _LIBCPP_HIDE_FROM_ABI result
  out(state_type& __st,
      const intern_type* __frm,
      const intern_type* __frm_end,
      const intern_type*& __frm_nxt,
      extern_type* __to,
      extern_type* __to_end,
      extern_type*& __to_nxt) const {
    return do_out(__st, __frm, __frm_end, __frm_nxt, __to, __to_end, __to_nxt);
  }

  _LIBCPP_HIDE_FROM_ABI result
  unshift(state_type& __st, extern_type* __to, extern_type* __to_end, extern_type*& __to_nxt) const {
    return do_unshift(__st, __to, __to_end, __to_nxt);
  }

  _LIBCPP_HIDE_FROM_ABI result
  in(state_type& __st,
     const extern_type* __frm,
     const extern_type* __frm_end,
     const extern_type*& __frm_nxt,
     intern_type* __to,
     intern_type* __to_end,
     intern_type*& __to_nxt) const {
    return do_in(__st, __frm, __frm_end, __frm_nxt, __to, __to_end, __to_nxt);
  }

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI int encoding() const _NOEXCEPT { return do_encoding(); }

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI bool always_noconv() const _NOEXCEPT { return do_always_noconv(); }

  _LIBCPP_HIDE_FROM_ABI int
  length(state_type& __st, const extern_type* __frm, const extern_type* __end, size_t __mx) const {
    return do_length(__st, __frm, __end, __mx);
  }

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI int max_length() const _NOEXCEPT { return do_max_length(); }

  static locale::id id;

protected:
  _LIBCPP_HIDE_FROM_ABI explicit codecvt(const char*, size_t __refs = 0) : locale::facet(__refs) {}

  ~codecvt() override;

  virtual result
  do_out(state_type& __st,
         const intern_type* __frm,
         const intern_type* __frm_end,
         const intern_type*& __frm_nxt,
         extern_type* __to,
         extern_type* __to_end,
         extern_type*& __to_nxt) const;
  virtual result
  do_in(state_type& __st,
        const extern_type* __frm,
        const extern_type* __frm_end,
        const extern_type*& __frm_nxt,
        intern_type* __to,
        intern_type* __to_end,
        intern_type*& __to_nxt) const;
  virtual result do_unshift(state_type& __st, extern_type* __to, extern_type* __to_end, extern_type*& __to_nxt) const;
  virtual int do_encoding() const _NOEXCEPT;
  virtual bool do_always_noconv() const _NOEXCEPT;
  virtual int do_length(state_type&, const extern_type* __frm, const extern_type* __end, size_t __mx) const;
  virtual int do_max_length() const _NOEXCEPT;
};

#  if _LIBCPP_HAS_CHAR8_T

// template <> class codecvt<char32_t, char8_t, mbstate_t> // C++20, deprecated in C++20

template <>
class _LIBCPP_DEPRECATED_IN_CXX20 _LIBCPP_EXPORTED_FROM_ABI codecvt<char32_t, char8_t, mbstate_t>
    : public locale::facet, public codecvt_base {
public:
  typedef char32_t intern_type;
  typedef char8_t extern_type;
  typedef mbstate_t state_type;

  _LIBCPP_HIDE_FROM_ABI explicit codecvt(size_t __refs = 0) : locale::facet(__refs) {}

  _LIBCPP_HIDE_FROM_ABI result
  out(state_type& __st,
      const intern_type* __frm,
      const intern_type* __frm_end,
      const intern_type*& __frm_nxt,
      extern_type* __to,
      extern_type* __to_end,
      extern_type*& __to_nxt) const {
    return do_out(__st, __frm, __frm_end, __frm_nxt, __to, __to_end, __to_nxt);
  }

  _LIBCPP_HIDE_FROM_ABI result
  unshift(state_type& __st, extern_type* __to, extern_type* __to_end, extern_type*& __to_nxt) const {
    return do_unshift(__st, __to, __to_end, __to_nxt);
  }

  _LIBCPP_HIDE_FROM_ABI result
  in(state_type& __st,
     const extern_type* __frm,
     const extern_type* __frm_end,
     const extern_type*& __frm_nxt,
     intern_type* __to,
     intern_type* __to_end,
     intern_type*& __to_nxt) const {
    return do_in(__st, __frm, __frm_end, __frm_nxt, __to, __to_end, __to_nxt);
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI int encoding() const _NOEXCEPT { return do_encoding(); }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI bool always_noconv() const _NOEXCEPT { return do_always_noconv(); }

  _LIBCPP_HIDE_FROM_ABI int
  length(state_type& __st, const extern_type* __frm, const extern_type* __end, size_t __mx) const {
    return do_length(__st, __frm, __end, __mx);
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI int max_length() const _NOEXCEPT { return do_max_length(); }

  static locale::id id;

protected:
  _LIBCPP_HIDE_FROM_ABI explicit codecvt(const char*, size_t __refs = 0) : locale::facet(__refs) {}

  ~codecvt() override;

  virtual result
  do_out(state_type& __st,
         const intern_type* __frm,
         const intern_type* __frm_end,
         const intern_type*& __frm_nxt,
         extern_type* __to,
         extern_type* __to_end,
         extern_type*& __to_nxt) const;
  virtual result
  do_in(state_type& __st,
        const extern_type* __frm,
        const extern_type* __frm_end,
        const extern_type*& __frm_nxt,
        intern_type* __to,
        intern_type* __to_end,
        intern_type*& __to_nxt) const;
  virtual result do_unshift(state_type& __st, extern_type* __to, extern_type* __to_end, extern_type*& __to_nxt) const;
  virtual int do_encoding() const _NOEXCEPT;
  virtual bool do_always_noconv() const _NOEXCEPT;
  virtual int do_length(state_type&, const extern_type* __frm, const extern_type* __end, size_t __mx) const;
  virtual int do_max_length() const _NOEXCEPT;
};

#  endif

// template <class _InternT, class _ExternT, class _StateT> class codecvt_byname

template <class _InternT, class _ExternT, class _StateT>
class codecvt_byname : public codecvt<_InternT, _ExternT, _StateT> {
public:
  _LIBCPP_HIDE_FROM_ABI explicit codecvt_byname(const char* __nm, size_t __refs = 0)
      : codecvt<_InternT, _ExternT, _StateT>(__nm, __refs) {}
  _LIBCPP_HIDE_FROM_ABI explicit codecvt_byname(const string& __nm, size_t __refs = 0)
      : codecvt<_InternT, _ExternT, _StateT>(__nm.c_str(), __refs) {}

protected:
  ~codecvt_byname() override;
};

_LIBCPP_SUPPRESS_DEPRECATED_PUSH
template <class _InternT, class _ExternT, class _StateT>
codecvt_byname<_InternT, _ExternT, _StateT>::~codecvt_byname() {}
_LIBCPP_SUPPRESS_DEPRECATED_POP

extern template class _LIBCPP_EXTERN_TEMPLATE_TYPE_VIS codecvt_byname<char, char, mbstate_t>;
#  if _LIBCPP_HAS_WIDE_CHARACTERS
extern template class _LIBCPP_EXTERN_TEMPLATE_TYPE_VIS codecvt_byname<wchar_t, char, mbstate_t>;
#  endif
extern template class _LIBCPP_DEPRECATED_IN_CXX20
_LIBCPP_EXTERN_TEMPLATE_TYPE_VIS codecvt_byname<char16_t, char, mbstate_t>; // deprecated in C++20
extern template class _LIBCPP_DEPRECATED_IN_CXX20
_LIBCPP_EXTERN_TEMPLATE_TYPE_VIS codecvt_byname<char32_t, char, mbstate_t>; // deprecated in C++20
#  if _LIBCPP_HAS_CHAR8_T
extern template class _LIBCPP_DEPRECATED_IN_CXX20
_LIBCPP_EXTERN_TEMPLATE_TYPE_VIS codecvt_byname<char16_t, char8_t, mbstate_t>; // C++20, deprecated in C++20
extern template class _LIBCPP_DEPRECATED_IN_CXX20
_LIBCPP_EXTERN_TEMPLATE_TYPE_VIS codecvt_byname<char32_t, char8_t, mbstate_t>; // C++20, deprecated in C++20
#  endif

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_HAS_LOCALIZATION

#endif // _LIBCPP___LOCALE_DIR_CODECVT_H
