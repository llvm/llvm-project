// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_STRING_MANAGER_H
#define _LIBCPP_STACKTRACE_STRING_MANAGER_H

#include <__config>
#include <ranges>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 23

#  include <__assert>
#  include <__cstddef/size_t.h>
#  include <__functional/function.h>
#  include <__fwd/istream.h>
#  include <__fwd/ostream.h>
#  include <__new/allocate.h>
#  include <__type_traits/is_allocator.h>
#  include <__vector/vector.h>
#  include <cstddef>
#  include <cstring>
#  include <memory>
#  include <string>
#  include <string_view>

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __stacktrace {

struct str {
  virtual ~str()                               = default;
  virtual str& reserve(size_t)                 = 0;
  virtual str& append(string_view)             = 0;
  virtual str& assign(string_view)             = 0;
  virtual str& clear()                         = 0;
  using overwrite_fn                           = function<size_t(char*, size_t)>;
  virtual str& overwrite(size_t, overwrite_fn) = 0;
  virtual str& getline(istream&)               = 0;
  virtual string_view view() const             = 0;

  size_t size() const { return view().size(); }
  char* data() { return const_cast<char*>(view().data()); }
  char const* data() const { return view().data(); }

  friend ostream& operator<<(ostream& __os, str const& __s) { return __os << __s.view(); }
};

struct literal_str final : str {
  char const* cstr_{};
  constexpr string_view view() const override { return {cstr_}; }
  str& reserve(size_t) override { return *this; }
  str& append(string_view) override { return *this; }
  str& assign(std::string_view) override { return *this; }
  str& clear() override { return *this; }
  str& overwrite(size_t __sz, std::function<size_t(char*, size_t)> __fn) override { return *this; }
  str& getline(istream&) override { return *this; }

  constexpr static literal_str const* empty() {
    constexpr static literal_str __ret;
    return &__ret;
  }
};

template < class _Alloc,
           class _ATraits = std::allocator_traits<_Alloc>,
           class _CAlloc  = _ATraits::template rebind_alloc<char>,
           class _BStr    = std::basic_string<char, std::char_traits<char>, _CAlloc>>
struct str_wrap final : _BStr, str {
  using _BStr::_BStr;

  std::string_view view() const override { return {*this}; }

  str& assign(std::string_view __view) override {
    _BStr::assign(__view);
    return *this;
  }
  str& append(std::string_view __view) override {
    _BStr::append(__view);
    return *this;
  }
  str& clear() override {
    _BStr::clear();
    return *this;
  }
  str& reserve(size_t __sz) override {
    _BStr::reserve(__sz);
    return *this;
  }
  str& overwrite(size_t __sz, std::function<size_t(char*, size_t)> __fn) override {
    _BStr::resize_and_overwrite(__sz, __fn);
    return *this;
  }
};

template <typename _Tp>
struct vec : ranges::view_interface<vec<_Tp>> {
  virtual ~vec()              = default;
  virtual _Tp& emplace_back() = 0;
  virtual _Tp* data()         = 0;
  virtual size_t size() const = 0;
  virtual bool empty() const  = 0;

  _Tp const* data() const { return const_cast<vec<_Tp>*>(this)->data(); }
  _Tp* begin() { return data(); }
  _Tp const* end() const { return data() + size(); }
};

template < class _Tp,
           class _Alloc,
           class _ATraits = std::allocator_traits<_Alloc>,
           class _TpAlloc = _ATraits::template rebind_alloc<_Tp>,
           class _Vec     = std::vector<_Tp, _TpAlloc>>
struct vec_wrap final : _Vec, vec<_Tp> {
  virtual ~vec_wrap() = default;
  _Tp* data() override { return _Vec::data(); }
  size_t size() const override { return _Vec::size(); }
  bool empty() const override { return _Vec::empty(); }
  _Tp& emplace_back() override { return _Vec::emplace_back(); }
};

template <size_t _Bytes>
struct fixedstr {
  char data_[_Bytes];
  size_t size_{0};

  size_t size() const { return size_; }
  bool empty() const { return !size(); }
  char* data() { return data_; }
  char const* data() const { return data_; }
  operator string_view() const { return {data(), size()}; }

  void assign(string_view __view) {
    strncpy(data_, __view.data(), sizeof(data_));
    size_ = strlen(data_);
  }
};

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_POP_MACROS

#endif // _LIBCPP_STACKTRACE_STRING_MANAGER_H
