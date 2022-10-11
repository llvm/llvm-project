//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_RESOURCE_MEMORY_RESOURCE_H
#define _LIBCPP___MEMORY_RESOURCE_MEMORY_RESOURCE_H

#include <__config>
#include <cstddef>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER > 14

_LIBCPP_BEGIN_NAMESPACE_STD

namespace pmr {

// [mem.res.class]

_LIBCPP_DIAGNOSTIC_PUSH
_LIBCPP_CLANG_DIAGNOSTIC_IGNORED("-Wweak-vtables") // TODO: move destructor into the dylib
class _LIBCPP_TYPE_VIS memory_resource {
  static const size_t __max_align = alignof(max_align_t);

public:
  virtual ~memory_resource() = default;

  _LIBCPP_HIDE_FROM_ABI void* allocate(size_t __bytes, size_t __align = __max_align) {
    return do_allocate(__bytes, __align);
  }

  _LIBCPP_HIDE_FROM_ABI void deallocate(void* __p, size_t __bytes, size_t __align = __max_align) {
    do_deallocate(__p, __bytes, __align);
  }

  _LIBCPP_HIDE_FROM_ABI bool is_equal(const memory_resource& __other) const noexcept { return do_is_equal(__other); }

private:
  virtual void* do_allocate(size_t, size_t)                       = 0;
  virtual void do_deallocate(void*, size_t, size_t)               = 0;
  virtual bool do_is_equal(memory_resource const&) const noexcept = 0;
};
_LIBCPP_DIAGNOSTIC_POP

// [mem.res.eq]

inline _LIBCPP_HIDE_FROM_ABI bool operator==(const memory_resource& __lhs, const memory_resource& __rhs) noexcept {
  return &__lhs == &__rhs || __lhs.is_equal(__rhs);
}

inline _LIBCPP_HIDE_FROM_ABI bool operator!=(const memory_resource& __lhs, const memory_resource& __rhs) noexcept {
  return !(__lhs == __rhs);
}

// [mem.res.global]

_LIBCPP_FUNC_VIS memory_resource* get_default_resource() noexcept;

_LIBCPP_FUNC_VIS memory_resource* set_default_resource(memory_resource*) noexcept;

_LIBCPP_FUNC_VIS memory_resource* new_delete_resource() noexcept;

_LIBCPP_FUNC_VIS memory_resource* null_memory_resource() noexcept;

} // namespace pmr

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER > 14

#endif // _LIBCPP___MEMORY_RESOURCE_MEMORY_RESOURCE_H
