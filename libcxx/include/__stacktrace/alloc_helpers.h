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
#  include <__type_traits/is_base_of.h>
#  include <__vector/vector.h>
#  include <cstddef>
#  include <cstring>
#  include <iostream>
#  include <memory>
#  include <string>
#  include <string_view>

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __stacktrace {

struct str;

using overwrite_fn _LIBCPP_NODEBUG = function<size_t(char*, size_t)>;

struct string_table_base {
  string_table_base()                                    = default;
  string_table_base(const string_table_base&)            = default;
  string_table_base(string_table_base&&)                 = default;
  string_table_base& operator=(const string_table_base&) = default;
  string_table_base& operator=(string_table_base&&)      = default;

  virtual ~string_table_base()                        = default;
  virtual str create()                                = 0;
  virtual string_view view(void*)                     = 0;
  virtual void reserve(void*, size_t)                 = 0;
  virtual void assign(void*, string_view)             = 0;
  virtual void overwrite(void*, size_t, overwrite_fn) = 0;
  virtual void getline(void* __index, istream& __is)  = 0;
  virtual void destroy(void*)                         = 0;
};

struct str final {
  string_table_base* __table_{};
  void* __str_ptr_{};

  ~str() {
    if (__table_ && __str_ptr_) {
      __table_->destroy(__str_ptr_);
    }
  }

  str()                      = default;
  str(const str&)            = default;
  str(str&&)                 = default;
  str& operator=(const str&) = default;
  str& operator=(str&&)      = default;
  str(string_table_base* __table, void* __str_ptr) : __table_(__table), __str_ptr_(__str_ptr) {}

  string_view view() const { return __str_ptr_ ? __table_->view(__str_ptr_) : string_view{}; }
  operator string_view() const { return view(); }

  size_t size() const { return view().size(); }
  bool empty() const { return !size(); }
  char* data() { return const_cast<char*>(view().data()); }
  char const* data() const { return const_cast<char*>(view().data()); }

  string_table_base& table() {
    _LIBCPP_ASSERT_NON_NULL(__table_, "no table associated with string");
    return *__table_;
  }

  void reserve(size_t __sz) { table().reserve(__str_ptr_, __sz); }
  void assign(string_view __sv) { table().assign(__str_ptr_, __sv); }
  void overwrite(size_t __sz, overwrite_fn __cb) { table().overwrite(__str_ptr_, __sz, __cb); }
  void getline(istream& __is) { table().getline(__str_ptr_, __is); }
};

template <class _A,
          class _AT = allocator_traits<_A>,
          class _CA = typename _AT::template rebind_alloc<char>,
          class _S  = basic_string<char, char_traits<char>, _CA>,
          class _SA = typename _AT::template rebind_alloc<_S>>
struct string_table : string_table_base {
  constexpr static void* kEmptyIndex = 0;

  ~string_table() override                     = default;
  string_table(const string_table&)            = default;
  string_table(string_table&&)                 = default;
  string_table& operator=(const string_table&) = default;
  string_table& operator=(string_table&&)      = default;

  [[no_unique_address]] _CA __char_alloc_;
  [[no_unique_address]] _SA __str_alloc_;

  explicit string_table(_A& __alloc) : __char_alloc_(_CA(__alloc)) {}

  str create() override {
    auto* __s = __str_alloc_.allocate(1);
    new (__s) _S(__char_alloc_);
    return str(this, __s);
  }

  void destroy(void* __p) override {
    auto* __s = (_S*)__p;
    __s->~_S();
    __str_alloc_.deallocate(__s, 1);
  }

  _S& at(void* __index) {
    _LIBCPP_ASSERT_NON_NULL(__index, "null basic_string ptr");
    return *(_S*)__index;
  }

  string_view view(void* __index) override { return at(__index); }

  void reserve(void* __index, size_t __sz) override { at(__index).reserve(__sz); }
  void assign(void* __index, string_view __sv) override { at(__index).assign(__sv); }
  void overwrite(void* __index, size_t __sz, overwrite_fn __cb) override {
    at(__index).resize_and_overwrite(__sz, __cb);
  }
  void getline(void* __index, istream& __is) override {
    __is.getline(const_cast<char*>(at(__index).data()), at(__index).capacity(), '\n');
  }
};

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_POP_MACROS

#endif // _LIBCPP_STACKTRACE_STRING_MANAGER_H
