//===-----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___LOCALE_LOCALE_H
#define _LIBCPP___LOCALE_LOCALE_H

#include <__config>
#include <__memory/shared_ptr.h>
#include <__mutex/once_flag.h>
#include <clocale>
#include <string>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

class _LIBCPP_EXPORTED_FROM_ABI locale;

template <class _CharT>
class _LIBCPP_TEMPLATE_VIS collate;

template <class _Facet>
_LIBCPP_HIDE_FROM_ABI bool has_facet(const locale&) _NOEXCEPT;

template <class _Facet>
_LIBCPP_HIDE_FROM_ABI const _Facet& use_facet(const locale&);

class _LIBCPP_EXPORTED_FROM_ABI locale {
public:
  // types:
  class _LIBCPP_EXPORTED_FROM_ABI facet;
  class _LIBCPP_EXPORTED_FROM_ABI id;

  typedef int category;

  static const category // values assigned here are for exposition only
      none    = 0,
      collate = LC_COLLATE_MASK, ctype = LC_CTYPE_MASK, monetary = LC_MONETARY_MASK, numeric = LC_NUMERIC_MASK,
      time = LC_TIME_MASK, messages = LC_MESSAGES_MASK, all = collate | ctype | monetary | numeric | time | messages;

  // construct/copy/destroy:
  locale() _NOEXCEPT;
  locale(const locale&) _NOEXCEPT;
  explicit locale(const char*);
  explicit locale(const string&);
  locale(const locale&, const char*, category);
  locale(const locale&, const string&, category);
  template <class _Facet>
  _LIBCPP_HIDE_FROM_ABI locale(const locale&, _Facet*);
  locale(const locale&, const locale&, category);

  ~locale();

  const locale& operator=(const locale&) _NOEXCEPT;

  template <class _Facet>
  _LIBCPP_METHOD_TEMPLATE_IMPLICIT_INSTANTIATION_VIS locale combine(const locale&) const;

  // locale operations:
  string name() const;
  bool operator==(const locale&) const;
#if _LIBCPP_STD_VER <= 17
  _LIBCPP_HIDE_FROM_ABI bool operator!=(const locale& __y) const { return !(*this == __y); }
#endif
  template <class _CharT, class _Traits, class _Allocator>
  _LIBCPP_METHOD_TEMPLATE_IMPLICIT_INSTANTIATION_VIS bool
  operator()(const basic_string<_CharT, _Traits, _Allocator>&, const basic_string<_CharT, _Traits, _Allocator>&) const;

  // global locale objects:
  static locale global(const locale&);
  static const locale& classic();

private:
  class __imp;
  __imp* __locale_;

  template <class>
  friend struct __no_destroy;
  struct __private_tag {};
  _LIBCPP_HIDE_FROM_ABI explicit locale(__private_tag, __imp* __loc) : __locale_(__loc) {}

  void __install_ctor(const locale&, facet*, long);
  static locale& __global();
  bool has_facet(id&) const;
  const facet* use_facet(id&) const;

  template <class _Facet>
  friend bool has_facet(const locale&) _NOEXCEPT;
  template <class _Facet>
  friend const _Facet& use_facet(const locale&);
};

class _LIBCPP_EXPORTED_FROM_ABI locale::facet : public __shared_count {
protected:
  _LIBCPP_HIDE_FROM_ABI explicit facet(size_t __refs = 0) : __shared_count(static_cast<long>(__refs) - 1) {}

  ~facet() override;

  //    facet(const facet&) = delete;     // effectively done in __shared_count
  //    void operator=(const facet&) = delete;

private:
  void __on_zero_shared() _NOEXCEPT override;
};

class _LIBCPP_EXPORTED_FROM_ABI locale::id {
  once_flag __flag_;
  int32_t __id_;

  static int32_t __next_id;

public:
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR id() : __id_(0) {}
  void operator=(const id&) = delete;
  id(const id&)             = delete;

public: // only needed for tests
  long __get();

  friend class locale;
  friend class locale::__imp;
};

template <class _Facet>
inline _LIBCPP_HIDE_FROM_ABI locale::locale(const locale& __other, _Facet* __f) {
  __install_ctor(__other, __f, __f ? __f->id.__get() : 0);
}

template <class _Facet>
locale locale::combine(const locale& __other) const {
  if (!std::has_facet<_Facet>(__other))
    __throw_runtime_error("locale::combine: locale missing facet");

  return locale(*this, &const_cast<_Facet&>(std::use_facet<_Facet>(__other)));
}

template <class _Facet>
inline _LIBCPP_HIDE_FROM_ABI bool has_facet(const locale& __l) _NOEXCEPT {
  return __l.has_facet(_Facet::id);
}

template <class _Facet>
inline _LIBCPP_HIDE_FROM_ABI const _Facet& use_facet(const locale& __l) {
  return static_cast<const _Facet&>(*__l.use_facet(_Facet::id));
}

template <class _CharT, class _Traits, class _Allocator>
bool locale::operator()(const basic_string<_CharT, _Traits, _Allocator>& __x,
                        const basic_string<_CharT, _Traits, _Allocator>& __y) const {
  return std::use_facet<std::collate<_CharT> >(*this).compare(
             __x.data(), __x.data() + __x.size(), __y.data(), __y.data() + __y.size()) < 0;
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___LOCALE_LOCALE_H
