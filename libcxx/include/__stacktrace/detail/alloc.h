// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_EXPERIMENTAL_STACKTRACE_ALLOC
#define _LIBCPP_EXPERIMENTAL_STACKTRACE_ALLOC

#include <__config>
#include <__functional/function.h>
#include <__memory/allocator_traits.h>
#include <__memory_resource/memory_resource.h>
#include <__memory_resource/polymorphic_allocator.h>
#include <cstddef>
#include <list>
#include <memory>
#include <string>

_LIBCPP_BEGIN_NAMESPACE_STD

class stacktrace_entry;

namespace __stacktrace {

/** Per-stacktrace-invocation allocator which wraps a caller-provided allocator of any type.
This is intended to be used with `std::pmr::` containers and strings throughout the stacktrace
creation process. */
struct alloc final : std::pmr::memory_resource {
  template <class _Allocator>
  _LIBCPP_HIDE_FROM_ABI explicit alloc(_Allocator const& __a) {
    // Take the given allocator type, and rebind with a new type having <byte> as the template arg
    using _AT       = std::allocator_traits<_Allocator>;
    using _BA       = typename _AT::template rebind_alloc<std::byte>;
    auto __ba       = _BA(__a);
    __alloc_func_   = [__ba](size_t __sz) mutable { return __ba.allocate(__sz); };
    __dealloc_func_ = [__ba](void* __ptr, size_t __sz) mutable { return __ba.deallocate((std::byte*)__ptr, __sz); };
    __alloc_opaque_ = std::addressof(__a);
  }

  _LIBCPP_HIDE_FROM_ABI_VIRTUAL ~alloc() override = default;

  _LIBCPP_HIDE_FROM_ABI_VIRTUAL virtual void _anchor_vfunc();

  _LIBCPP_HIDE_FROM_ABI_VIRTUAL void* do_allocate(size_t __size, size_t __align) override {
    // Avoiding "assert" in a system header, but we expect this to hold:
    // assert(__align <= alignof(std::max_align_t));
    (void)__align;
    return __alloc_func_(__size);
  }

  _LIBCPP_HIDE_FROM_ABI_VIRTUAL void do_deallocate(void* __ptr, size_t __size, size_t __align) override {
    (void)__align;
    __dealloc_func_((std::byte*)__ptr, __size);
  }

  _LIBCPP_HIDE_FROM_ABI_VIRTUAL bool do_is_equal(std::pmr::memory_resource const& __rhs) const noexcept override {
    auto* __rhs_ba = dynamic_cast<alloc const*>(&__rhs);
    return __rhs_ba && (__rhs_ba->__alloc_opaque_ == __alloc_opaque_);
  }

  _LIBCPP_HIDE_FROM_ABI std::pmr::string new_string(size_t __size = 0) {
    std::pmr::string __ret{this};
    if (__size) {
      __ret.reserve(__size);
      __ret[0] = 0;
    }
    return __ret;
  }

  _LIBCPP_HIDE_FROM_ABI std::pmr::string hex_string(uintptr_t __addr) {
    char __ret[19]; // "0x" + 16 digits + NUL
    auto __size = snprintf(__ret, sizeof(__ret), "0x%016llx", (unsigned long long)__addr);
    return {__ret, size_t(__size), this};
  }

  _LIBCPP_HIDE_FROM_ABI std::pmr::string u64_string(uintptr_t __val) {
    char __ret[21]; // 20 digits max + NUL
    auto __size = snprintf(__ret, sizeof(__ret), "%zu", __val);
    return {__ret, size_t(__size), this};
  }

  template <typename _Tp>
  _LIBCPP_HIDE_FROM_ABI std::pmr::list<_Tp> new_list() {
    return std::pmr::list<_Tp>{this};
  }

  _LIBCPP_HIDE_FROM_ABI std::pmr::list<std::pmr::string> new_string_list() { return new_list<std::pmr::string>(); }

private:
  std::function<std::byte*(size_t)> __alloc_func_;
  std::function<void(std::byte*, size_t)> __dealloc_func_;
  /** Only used for equality */
  void const* __alloc_opaque_;
};

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_EXPERIMENTAL_STACKTRACE_ALLOC
