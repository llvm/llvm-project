// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_BASIC
#define _LIBCPP_STACKTRACE_BASIC

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
#  include <__functional/hash.h>
#  include <__fwd/format.h>
#  include <__fwd/ostream.h>
#  include <__iterator/iterator.h>
#  include <__iterator/reverse_iterator.h>
#  include <__memory/allocator_traits.h>
#  include <__memory_resource/polymorphic_allocator.h>
#  include <__new/allocate.h>
#  include <__type_traits/is_nothrow_constructible.h>
#  include <__vector/vector.h>
#  include <cstddef>
#  include <cstdint>
#  include <string>
#  include <type_traits>
#  include <utility>

#  include <__stacktrace/memory.h>
#  include <__stacktrace/stacktrace_entry.h>

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __stacktrace {

struct base {
  constexpr static size_t __default_max_depth    = 64;
  constexpr static size_t __absolute_max_depth   = 256;
  constexpr static size_t __k_init_pool_on_stack = 1 << 12;

  std::function<size_t()> __entries_size_;
  std::function<entry_base&()> __emplace_entry_;
  std::function<entry_base*()> __entries_data_;
  std::function<entry_base&(size_t)> __entry_at_;

  template <class _Vp>
  _LIBCPP_HIDE_FROM_ABI base(_Vp* __entries)
      : __entries_size_([=]() { return __entries->size(); }),
        __emplace_entry_([=]() -> entry_base& { return (entry_base&)__entries->emplace_back(); }),
        __entries_data_([=]() -> entry_base* { return (entry_base*)__entries->data(); }),
        __entry_at_([=](size_t __i) -> entry_base& { return (entry_base&)__entries->at(__i); }) {}

  _LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE _LIBCPP_EXPORTED_FROM_ABI void
  current(arena& __arena, size_t __skip, size_t __max_depth);

  _LIBCPP_HIDE_FROM_ABI void find_images(arena& __arena);
  _LIBCPP_HIDE_FROM_ABI void find_symbols(arena& __arena);
  _LIBCPP_HIDE_FROM_ABI void find_source_locs(arena& __arena);

  _LIBCPP_EXPORTED_FROM_ABI entry_base* entries_begin() { return __entries_data_(); }
  _LIBCPP_EXPORTED_FROM_ABI entry_base* entries_end() { return __entries_data_() + __entries_size_(); }

  _LIBCPP_EXPORTED_FROM_ABI std::ostream& write_to(std::ostream& __os) const;
  _LIBCPP_EXPORTED_FROM_ABI string to_string() const;
};

} // namespace __stacktrace

// (19.6.4)
// Class template basic_stacktrace [stacktrace.basic]

class stacktrace_entry;

template <class _Allocator>
class basic_stacktrace : private __stacktrace::base {
  friend struct hash<basic_stacktrace<_Allocator>>;

  using _ATraits _LIBCPP_NODEBUG            = allocator_traits<_Allocator>;
  constexpr static bool __kPropOnCopyAssign = _ATraits::propagate_on_container_copy_assignment::value;
  constexpr static bool __kPropOnMoveAssign = _ATraits::propagate_on_container_move_assignment::value;
  constexpr static bool __kPropOnSwap       = _ATraits::propagate_on_container_swap::value;
  constexpr static bool __kAlwaysEqual      = _ATraits::is_always_equal::value;
  constexpr static bool __kNoThrowAlloc =
      noexcept(noexcept(_Allocator().allocate(1)) && noexcept(_Allocator().allocate_at_least(1)));

  [[no_unique_address]]
  _Allocator __alloc_;

  using entry_vec _LIBCPP_NODEBUG = std::vector<stacktrace_entry, _Allocator>;
  entry_vec __entries_;

public:
  // (19.6.4.1)
  // Overview [stacktrace.basic.overview]

  using value_type             = stacktrace_entry;
  using const_reference        = const value_type&;
  using reference              = value_type&;
  using difference_type        = ptrdiff_t;
  using size_type              = size_t;
  using allocator_type         = _Allocator;
  using const_iterator         = entry_vec::const_iterator;
  using iterator               = const_iterator;
  using reverse_iterator       = std::reverse_iterator<basic_stacktrace::iterator>;
  using const_reverse_iterator = std::reverse_iterator<basic_stacktrace::const_iterator>;

  // (19.6.4.2)
  // Creation and assignment [stacktrace.basic.cons]

  _LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE _LIBCPP_EXPORTED_FROM_ABI static basic_stacktrace
  current(const allocator_type& __caller_alloc = allocator_type()) noexcept(__kNoThrowAlloc) {
    size_type __skip      = 1;
    size_type __max_depth = __default_max_depth;
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        __skip <= __skip + __max_depth, "sum of skip and max_depth overflows size_type");
    basic_stacktrace __ret{__caller_alloc};
    __stacktrace::stack_bytes<__k_init_pool_on_stack> __stack_bytes;
    __stacktrace::byte_pool __stack_pool = __stack_bytes.pool();
    __stacktrace::arena __arena(__stack_pool, __caller_alloc);
    ((base&)__ret).current(__arena, __skip, __max_depth);
    return __ret;
  }

  _LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE _LIBCPP_EXPORTED_FROM_ABI static basic_stacktrace
  current(size_type __skip, const allocator_type& __caller_alloc = allocator_type()) noexcept(__kNoThrowAlloc) {
    ++__skip;
    size_type __max_depth = __default_max_depth;
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        __skip <= __skip + __max_depth, "sum of skip and max_depth overflows size_type");
    basic_stacktrace __ret{__caller_alloc};
    __stacktrace::stack_bytes<__k_init_pool_on_stack> __stack_bytes;
    __stacktrace::byte_pool __stack_pool = __stack_bytes.pool();
    __stacktrace::arena __arena(__stack_pool, __caller_alloc);
    ((base&)__ret).current(__arena, __skip, __max_depth);
    return __ret;
  }

  _LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE _LIBCPP_EXPORTED_FROM_ABI static basic_stacktrace
  current(size_type __skip,
          size_type __max_depth,
          const allocator_type& __caller_alloc = allocator_type()) noexcept(__kNoThrowAlloc) {
    ++__skip;
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        __skip <= __skip + __max_depth, "sum of skip and max_depth overflows size_type");
    basic_stacktrace __ret{__caller_alloc};
    if (__max_depth) [[likely]] {
      __stacktrace::stack_bytes<__k_init_pool_on_stack> __stack_bytes;
      __stacktrace::byte_pool __stack_pool = __stack_bytes.pool();
      __stacktrace::arena __arena(__stack_pool, __caller_alloc);
      ((base&)__ret).current(__arena, __skip, __max_depth);
    }
    return __ret;
  }

  _LIBCPP_EXPORTED_FROM_ABI constexpr ~basic_stacktrace() = default;

  _LIBCPP_EXPORTED_FROM_ABI explicit basic_stacktrace(const allocator_type& __alloc)
      : base(&__entries_), __alloc_(__alloc), __entries_(__alloc_) {}

  _LIBCPP_EXPORTED_FROM_ABI basic_stacktrace(basic_stacktrace const& __other, allocator_type const& __alloc)
      : base(&__entries_), __alloc_(__alloc), __entries_(__other.__entries_) {}

  _LIBCPP_EXPORTED_FROM_ABI basic_stacktrace(basic_stacktrace&& __other, allocator_type const& __alloc)
      : base(&__entries_), __alloc_(__alloc), __entries_(std::move(__other.__entries_)) {}

  _LIBCPP_EXPORTED_FROM_ABI basic_stacktrace() noexcept(is_nothrow_default_constructible_v<allocator_type>)
      : basic_stacktrace(allocator_type()) {}

  _LIBCPP_EXPORTED_FROM_ABI basic_stacktrace(basic_stacktrace const& __other) noexcept
      : basic_stacktrace(__other, _ATraits::select_on_container_copy_construction(__other.__alloc_)) {}

  _LIBCPP_EXPORTED_FROM_ABI basic_stacktrace(basic_stacktrace&& __other) noexcept
      : basic_stacktrace(std::move(__other), __other.__alloc_) {}

  _LIBCPP_EXPORTED_FROM_ABI basic_stacktrace& operator=(const basic_stacktrace& __other) {
    if (std::addressof(__other) != this) {
      if (__kPropOnCopyAssign) {
        new (this) basic_stacktrace(__other, __other.__alloc_);
      } else {
        new (this) basic_stacktrace(__other);
      }
    }
    return *this;
  }

  _LIBCPP_EXPORTED_FROM_ABI basic_stacktrace&
  operator=(basic_stacktrace&& __other) noexcept(__kPropOnMoveAssign || __kAlwaysEqual) {
    if (std::addressof(__other) != this) {
      if (__kPropOnMoveAssign) {
        auto __alloc = __other.__alloc_;
        new (this) basic_stacktrace(std::move(__other), __alloc);
      } else {
        new (this) basic_stacktrace(std::move(__other));
      }
    }
    return *this;
  }

  // (19.6.4.3)
  // [stacktrace.basic.obs], observers

  [[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI allocator_type get_allocator() const noexcept { return __alloc_; }

  [[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI const_iterator begin() const noexcept { return __entries_.begin(); }
  [[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI const_iterator end() const noexcept { return __entries_.end(); }
  [[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI const_reverse_iterator rbegin() const noexcept { return __entries_.rbegin(); }
  [[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI const_reverse_iterator rend() const noexcept { return __entries_.rend(); }

  [[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI const_iterator cbegin() const noexcept { return __entries_.cbegin(); }
  [[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI const_iterator cend() const noexcept { return __entries_.cend(); }
  [[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI const_reverse_iterator crbegin() const noexcept {
    return __entries_.crbegin();
  }
  [[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI const_reverse_iterator crend() const noexcept { return __entries_.crend(); }

  [[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI bool empty() const noexcept { return __entries_.empty(); }
  [[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI size_type size() const noexcept { return __entries_.size(); }
  [[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI size_type max_size() const noexcept { return __entries_.max_size(); }

  [[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI const_reference operator[](size_type __i) const { return __entries_[__i]; }
  [[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI const_reference at(size_type __i) const { return __entries_.at(__i); }

  // (19.6.4.4)
  // [stacktrace.basic.cmp], comparisons

  template <class _Allocator2>
  _LIBCPP_EXPORTED_FROM_ABI friend bool
  operator==(const basic_stacktrace& __x, const basic_stacktrace<_Allocator2>& __y) noexcept {
    if (__x.size() != __y.size()) {
      return false;
    }
    auto __xi = __x.begin();
    auto __yi = __y.begin();
    auto __xe = __x.end();
    while (__xi != __xe) {
      if (*__xi++ != *__yi++) {
        return false;
      }
    }
    return true;
  }

  template <class _Allocator2>
  _LIBCPP_EXPORTED_FROM_ABI friend strong_ordering
  operator<=>(const basic_stacktrace& __x, const basic_stacktrace<_Allocator2>& __y) noexcept {
    auto __ret = __x.size() <=> __y.size();
    if (__ret != std::strong_ordering::equal) {
      return __ret;
    }
    auto __xi = __x.begin();
    auto __yi = __y.begin();
    auto __xe = __x.end();
    while ((__ret == std::strong_ordering::equal) && __xi != __xe) {
      __ret = *__xi++ <=> *__yi++;
    }
    return __ret;
  }

  // (19.6.4.5)
  // [stacktrace.basic.mod], modifiers

  _LIBCPP_EXPORTED_FROM_ABI void swap(basic_stacktrace& __other) noexcept(
      allocator_traits<_Allocator>::propagate_on_container_swap::value ||
      allocator_traits<_Allocator>::is_always_equal::value) {
    std::swap(__entries_, __other.__entries_);
    if (__kPropOnSwap) {
      std::swap(__alloc_, __other.__alloc_);
    }
  }
};

using stacktrace = basic_stacktrace<allocator<stacktrace_entry>>;

namespace pmr {
using stacktrace = basic_stacktrace<polymorphic_allocator<stacktrace_entry>>;
} // namespace pmr

// (19.6.4.6)
// Non-member functions [stacktrace.basic.nonmem]

template <class _Allocator>
_LIBCPP_EXPORTED_FROM_ABI inline void
swap(basic_stacktrace<_Allocator>& __a, basic_stacktrace<_Allocator>& __b) noexcept(noexcept(__a.swap(__b))) {
  __a.swap(__b);
}

template <class _Allocator>
_LIBCPP_EXPORTED_FROM_ABI ostream& operator<<(ostream& __os, const basic_stacktrace<_Allocator>& __stacktrace) {
  return ((__stacktrace::base const&)__stacktrace).write_to(__os);
}

template <class _Allocator>
_LIBCPP_EXPORTED_FROM_ABI string to_string(const basic_stacktrace<_Allocator>& __stacktrace) {
  return ((__stacktrace::base const&)__stacktrace).to_string();
}

// (19.6.6)
// Hash support [stacktrace.basic.hash]

template <class _Allocator>
struct hash<basic_stacktrace<_Allocator>> {
  [[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI size_t
  operator()(basic_stacktrace<_Allocator> const& __context) const noexcept {
    size_t __ret = 1;
    for (auto const& __entry : __context.__entries_) {
      __ret += hash<uintptr_t>()(__entry.native_handle());
    }
    return __ret;
  }
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_POP_MACROS

#endif // _LIBCPP_STACKTRACE_BASIC
