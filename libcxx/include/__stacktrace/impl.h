// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_IMPL
#define _LIBCPP_STACKTRACE_IMPL

#include <__cstddef/byte.h>
#include <__cstddef/ptrdiff_t.h>
#include <__cstddef/size_t.h>
#include <__format/formatter.h>
#include <__functional/function.h>
#include <__functional/hash.h>
#include <__fwd/format.h>
#include <__fwd/ostream.h>
#include <__fwd/sstream.h>
#include <__fwd/vector.h>
#include <__iterator/iterator.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/reverse_access.h>
#include <__iterator/reverse_iterator.h>
#include <__memory/allocator.h>
#include <__memory/allocator_traits.h>
#include <__memory_resource/memory_resource.h>
#include <__memory_resource/polymorphic_allocator.h>
#include <__utility/move.h>
#include <__vector/pmr.h>
#include <__vector/swap.h>
#include <__vector/vector.h>
#include <list>
#include <string>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Allocator>
class basic_stacktrace;

class stacktrace_entry;

namespace __stacktrace {

/** Per-stacktrace-invocation allocator which wraps a caller-provided allocator of any type.
This is intended to be used with `std::pmr::` containers and strings throughout the stacktrace
creation process. */
struct _LIBCPP_HIDE_FROM_ABI alloc final : std::pmr::memory_resource {
  template <class _Allocator>
  explicit alloc(_Allocator const& __a) {
    // Take the given allocator type, and rebind with a new type having <byte> as the template arg
    using _AT       = std::allocator_traits<_Allocator>;
    using _BA       = typename _AT::template rebind_alloc<std::byte>;
    auto __ba       = _BA(__a);
    __alloc_func_   = [__ba](size_t __sz) mutable { return __ba.allocate(__sz); };
    __dealloc_func_ = [__ba](void* __ptr, size_t __sz) mutable { return __ba.deallocate((std::byte*)__ptr, __sz); };
    __alloc_ptr_    = std::addressof(__a);
  }

  _LIBCPP_HIDE_FROM_ABI ~alloc() override = default;

  _LIBCPP_HIDE_FROM_ABI void* do_allocate(size_t __size, size_t /*__align*/) override { return __alloc_func_(__size); }

  _LIBCPP_HIDE_FROM_ABI void do_deallocate(void* __ptr, size_t __size, size_t /*__align*/) override {
    __dealloc_func_((std::byte*)__ptr, __size);
  }

  _LIBCPP_HIDE_FROM_ABI bool do_is_equal(std::pmr::memory_resource const& __rhs) const noexcept override {
    auto* __rhs_ba = dynamic_cast<alloc const*>(&__rhs);
    return __rhs_ba && (__rhs_ba->__alloc_ptr_ == __alloc_ptr_);
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

  std::pmr::list<std::pmr::string> new_string_list() { return new_list<std::pmr::string>(); }

private:
  std::function<std::byte*(size_t)> __alloc_func_;
  std::function<void(std::byte*, size_t)> __dealloc_func_;
  /** Only used for checking equality */
  void const* __alloc_ptr_;
};

/** Contains fields which will be used to generate the final `std::stacktrace_entry`.
This is an intermediate object which owns strings allocated via the caller-provided allocator,
which are later freed back to that allocator and converted to plain `std::string`s. */
struct entry {
  /** Caller's / faulting insn's address, including ASLR/slide */
  uintptr_t __addr_{};

  /** the address minus its image's slide offset */
  uintptr_t __addr_unslid_{};

  /** entry's description (symbol name) */
  std::pmr::string __desc_{};

  /** source file name */
  std::pmr::string __file_{};

  /** line number in source file */
  uint32_t __line_{};

  /* implicit */ operator std::stacktrace_entry();
};

struct __to_string {
  _LIBCPP_EXPORTED_FROM_ABI string operator()(stacktrace_entry const& __entry);

  _LIBCPP_EXPORTED_FROM_ABI void operator()(ostream& __os, stacktrace_entry const& __entry);

  _LIBCPP_EXPORTED_FROM_ABI void operator()(ostream& __os, std::stacktrace_entry const* __entries, size_t __count);

  _LIBCPP_EXPORTED_FROM_ABI string operator()(std::stacktrace_entry const* __entries, size_t __count);

  template <class _Allocator>
  _LIBCPP_EXPORTED_FROM_ABI string operator()(basic_stacktrace<_Allocator> const& __st) {
    return (*this)(__st.__entries_.data(), __st.__entries_.size());
  }
};

_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE _LIBCPP_EXPORTED_FROM_ABI void
__impl(size_t __skip,
       size_t __max_depth,
       alloc& __alloc,
       std::function<void(size_t)> __resize_func,
       std::function<void(size_t, std::stacktrace_entry&&)> __assign_func);

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_IMPL
