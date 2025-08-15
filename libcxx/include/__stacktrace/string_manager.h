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
#include <memory>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 23

#  include <__assert>
#  include <__cstddef/size_t.h>
#  include <__functional/function.h>
#  include <__new/allocate.h>
#  include <__vector/vector.h>
#  include <cstddef>
#  include <string>

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __stacktrace {

template <typename _Tp>
struct alloc {
  using value_type                             = _Tp;
  using pointer                                = _Tp*;
  using is_always_equal                        = std::false_type;
  using propagate_on_container_copy_assignment = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap            = std::true_type;

  template <typename _Tp2>
  struct rebind {
    using other = alloc<_Tp2>;
  };

  [[no_unique_address]] function<char*(size_t)> __alloc_;
  [[no_unique_address]] function<void(char*, size_t)> __dealloc_;

  template <class _Alloc>
    requires __is_allocator<_Alloc>::value
  explicit alloc(_Alloc __alloc) {
    using _ATraits   = allocator_traits<_Alloc>;
    using _CharAlloc = _ATraits::template rebind_alloc<char>;
    _CharAlloc __ca{__alloc};
    __alloc_   = [__ca](size_t __sz) mutable { return __ca.allocate(__sz); };
    __dealloc_ = [__ca](char* __p, size_t __sz) mutable { __ca.deallocate(__p, __sz); };
  }

  alloc(const alloc&)            = default;
  alloc(alloc&&)                 = default;
  alloc& operator=(const alloc&) = default;
  alloc& operator=(alloc&&)      = default;

  bool operator==(alloc const& __rhs) const { return std::addressof(__rhs) == this; }
  _Tp* allocate(size_t __sz) { return __alloc_(__sz); }
  void deallocate(_Tp* __p, size_t __sz) { __dealloc_(__p, __sz); }
};

using str = basic_string<char, char_traits<char>, alloc<char>>;

struct string_manager {
  alloc<char> __char_alloc_;

  string_manager(const string_manager&)            = default;
  string_manager& operator=(const string_manager&) = default;

  str make_str() { return str{__char_alloc_}; }

  str make_str(string_view __view) {
    auto __ret = make_str();
    __ret.assign(__view);
    return __ret;
  }

  // Uses the allocator provided the to `current` call
  string_manager(auto const& __alloc) : __char_alloc_(__alloc) {}
};

// template <typename _Tp>
// struct alloc {
//   using value_type = _Tp;
//   using pointer    = _Tp*;
//   template <typename _Tp2>
//   struct rebind {
//     using other = alloc<_Tp2>;
//   };
//   _Tp* allocate(size_t __sz) { return new _Tp[__sz]; }
//   void deallocate(_Tp* __p, size_t __sz) {}
//   bool operator==(alloc const& __rhs) const { return std::addressof(__rhs) == this; }
// };

// struct str : basic_string<char, char_traits<char>, alloc<char>> {
//   //   size_t __index_{0}; // default to index 0 which is for the empty, dummy string
//   //   void assign(string_view __view) { string_manager_base::get().assign(__index_, __view); }
//   //   void resize(size_t __size) { string_manager_base::get().resize(__index_, __size); }
//   //   void getline(istream& __is) { string_manager_base::get().getline(__index_, __is); }
//   //   string_view view() const { return string_manager_base::get().view(__index_); }
//   //   operator string_view() const { return view(); }
//   //   char* data() { return const_cast<char*>(view().data()); }
//   //   size_t size() const { return view().size(); }

//   operator bool() const { return size(); }

//   template <typename... _AL>
//   _LIBCPP_HIDE_FROM_ABI str& makef(char const* __fmt, _AL&&... __args) {
// #  ifdef __clang__
// #    pragma clang diagnostic push
// #    pragma clang diagnostic ignored "-Wformat-security"
// #    pragma clang diagnostic ignored "-Wformat-nonliteral"
// #  endif
//     auto __size = 1 + std::snprintf(nullptr, 0, __fmt, __args...);
//     resize(__size);
//     std::snprintf(data(), __size, __fmt, std::forward<_AL>(__args)...);
// #  ifdef __clang__
// #    pragma clang diagnostic pop
// #  endif
//     return *this;
//   }
// };

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_POP_MACROS

#endif // _LIBCPP_STACKTRACE_STRING_MANAGER_H
