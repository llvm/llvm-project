// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_BASE
#define _LIBCPP_STACKTRACE_BASE

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 23

#  include <__cstddef/byte.h>
#  include <__cstddef/size_t.h>
#  include <__functional/function.h>
#  include <__fwd/format.h>
#  include <__fwd/ostream.h>
#  include <__memory/allocator.h>
#  include <__memory/allocator_traits.h>
#  include <__new/allocate.h>
#  include <__vector/vector.h>
#  include <cstddef>
#  include <cstdint>
#  include <list>
#  include <optional>
#  include <string>

_LIBCPP_BEGIN_NAMESPACE_STD

class _LIBCPP_EXPORTED_FROM_ABI stacktrace_entry;

namespace __stacktrace {

struct _LIBCPP_HIDE_FROM_ABI entry_base;

struct _LIBCPP_EXPORTED_FROM_ABI base {
  template <typename _Tp>
  struct _LIBCPP_HIDE_FROM_ABI Alloc {
    function<byte*(size_t)> __alloc_bytes_;
    function<void(byte*, size_t)> __dealloc_bytes_;

    Alloc(function<byte*(size_t)> __alloc_bytes, function<void(byte*, size_t)> __dealloc_bytes)
        : __alloc_bytes_(__alloc_bytes), __dealloc_bytes_(__dealloc_bytes) {}

    template <typename _T2 = _Tp>
    Alloc(Alloc<_T2> const& __rhs) : Alloc(__rhs.__alloc_bytes_, __rhs.__dealloc_bytes_) {}

    Alloc()
        : __alloc_bytes_([](size_t __sz) { return std::allocator<std::byte>().allocate(__sz); }),
          __dealloc_bytes_([](std::byte* __ptr, size_t __sz) { std::allocator<std::byte>().deallocate(__ptr, __sz); }) {
    }

    // XXX Alignment?
    using value_type = _Tp;
    [[nodiscard]] _Tp* allocate(size_t __sz) { return (_Tp*)__alloc_bytes_(__sz * sizeof(_Tp)); }
    void deallocate(_Tp* __ptr, size_t __sz) { __dealloc_bytes_((byte*)__ptr, __sz * sizeof(_Tp)); }

    template <typename _A2>
    bool operator==(_A2 const& __rhs) const {
      return std::addressof(__rhs) == this;
    }
  };

  template <typename _Tp>
  Alloc<_Tp> _LIBCPP_HIDE_FROM_ABI make_alloc() {
    return {__alloc_bytes_, __dealloc_bytes_};
  }

  using str = basic_string<char, char_traits<char>, Alloc<char>>;

  template <typename... _Args>
  str _LIBCPP_HIDE_FROM_ABI make_str(_Args... __args) {
    return str(std::forward<_Args>(__args)..., make_alloc<char>());
  }

  template <typename _Tp>
  using vec = vector<_Tp, Alloc<_Tp>>;

  template <typename _Tp, typename... _Args>
  _LIBCPP_HIDE_FROM_ABI vec<_Tp> make_vec(_Args... __args) {
    return vec(std::forward<_Args>(__args)..., make_alloc<_Tp>());
  }

  template <typename _Tp>
  using list = ::std::list<_Tp, Alloc<_Tp>>;

  template <typename _Tp, typename... _Args>
  _LIBCPP_HIDE_FROM_ABI list<_Tp> make_list(_Args... __args) {
    return list(std::forward<_Args>(__args)..., make_alloc<_Tp>());
  }

  template <class _Allocator>
  auto _LIBCPP_HIDE_FROM_ABI __alloc_wrap(_Allocator const& __alloc) {
    using _AT = allocator_traits<_Allocator>;
    using _BA = typename _AT::template rebind_alloc<byte>;
    auto __ba = _BA(__alloc);
    return [__ba = std::move(__ba)](size_t __sz) mutable { return __ba.allocate(__sz); };
  }

  template <class _Allocator>
  auto _LIBCPP_HIDE_FROM_ABI __dealloc_wrap(_Allocator const& __alloc) {
    using _AT = allocator_traits<_Allocator>;
    using _BA = typename _AT::template rebind_alloc<byte>;
    auto __ba = _BA(__alloc);
    return [__ba = std::move(__ba)](void* __ptr, size_t __sz) mutable { __ba.deallocate((byte*)__ptr, __sz); };
  }

  _LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE _LIBCPP_EXPORTED_FROM_ABI void
  build_stacktrace(size_t __skip, size_t __max_depth);

  base();

  template <class _Allocator>
  explicit _LIBCPP_EXPORTED_FROM_ABI base(_Allocator __alloc);

  function<byte*(size_t)> __alloc_bytes_;
  function<void(byte*, size_t)> __dealloc_bytes_;
  vec<entry_base> __entries_;
  str __main_prog_path_;
};

struct _LIBCPP_HIDE_FROM_ABI entry_base {
  uintptr_t __addr_actual_{};                  // this address, as observed in this current process
  uintptr_t __addr_unslid_{};                  // address adjusted for ASLR
  optional<__stacktrace::base::str> __desc_{}; // uses wrapped _Allocator from caller
  optional<__stacktrace::base::str> __file_{}; // uses wrapped _Allocator from caller
  uint_least32_t __line_{};

  _LIBCPP_HIDE_FROM_ABI stacktrace_entry to_stacktrace_entry() const;
};

template <class _Allocator>
_LIBCPP_EXPORTED_FROM_ABI base::base(_Allocator __alloc)
    : __alloc_bytes_(__alloc_wrap(__alloc)),
      __dealloc_bytes_(__dealloc_wrap(__alloc)),
      __entries_(make_vec<entry_base>()),
      __main_prog_path_(make_str()) {}

} // namespace __stacktrace

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_POP_MACROS

#endif // _LIBCPP_STACKTRACE_BASE
