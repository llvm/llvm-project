// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_BASIC_STACKTRACE
#define _LIBCPP_BASIC_STACKTRACE

#include <__config>
#include <__functional/hash.h>
#include <__fwd/format.h>
#include <__iterator/iterator.h>
#include <__iterator/reverse_iterator.h>
#include <__memory/allocator_traits.h>
#include <__memory_resource/polymorphic_allocator.h>
#include <__stacktrace/base.h>
#include <__stacktrace/entry.h>
#include <__stacktrace/to_string.h>
#include <__type_traits/is_nothrow_constructible.h>
#include <__vector/vector.h>
#include <utility>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

// (19.6.4)
// Class template basic_stacktrace [stacktrace.basic]

class stacktrace_entry;

template <class _Allocator>
class _LIBCPP_EXPORTED_FROM_ABI basic_stacktrace {
  friend struct hash<basic_stacktrace<_Allocator>>;
  friend struct __stacktrace::__to_string;

  using _ATraits _LIBCPP_NODEBUG               = allocator_traits<_Allocator>;
  constexpr static bool __kPropOnCopy          = _ATraits::propagate_on_container_copy_assignment::value;
  constexpr static bool __kPropOnMove          = _ATraits::propagate_on_container_move_assignment::value;
  constexpr static bool __kPropOnSwap          = _ATraits::propagate_on_container_swap::value;
  constexpr static bool __kAlwaysEqual         = _ATraits::is_always_equal::value;
  constexpr static bool __kNoThrowDflConstruct = is_nothrow_default_constructible_v<_Allocator>;
  constexpr static bool __kNoThrowAlloc =
      noexcept(noexcept(_Allocator().allocate(1)) && noexcept(_Allocator().allocate_at_least(1)));

  [[no_unique_address]] _Allocator __alloc_;

  using __entry_vec = vector<stacktrace_entry, _Allocator>;
  __entry_vec __entries_;

public:
  // (19.6.4.1)
  // Overview [stacktrace.basic.overview]

  using value_type      = stacktrace_entry;
  using const_reference = const value_type&;
  using reference       = value_type&;
  using difference_type = ptrdiff_t;
  using size_type       = size_t;
  using allocator_type  = _Allocator;
  using const_iterator  = __entry_vec::const_iterator;
  using iterator        = const_iterator;

  using reverse_iterator       = std::reverse_iterator<basic_stacktrace::iterator>;
  using const_reverse_iterator = std::reverse_iterator<basic_stacktrace::const_iterator>;

  // (19.6.4.2)
  // Creation and assignment [stacktrace.basic.cons]

  _LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE _LIBCPP_EXPORTED_FROM_ABI static basic_stacktrace
  current(const allocator_type& __caller_alloc = allocator_type()) noexcept(__kNoThrowAlloc) {
    return current(1, /* no __max_depth */ ~0, __caller_alloc);
  }

  _LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE _LIBCPP_EXPORTED_FROM_ABI static basic_stacktrace
  current(size_type __skip, const allocator_type& __caller_alloc = allocator_type()) noexcept(__kNoThrowAlloc) {
    return current(__skip + 1, /* no __max_depth */ ~0, __caller_alloc);
  }

  _LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE _LIBCPP_EXPORTED_FROM_ABI static basic_stacktrace
  current(size_type __skip,
          size_type __max_depth,
          const allocator_type& __caller_alloc = allocator_type()) noexcept(__kNoThrowAlloc) {
    __stacktrace::builder __builder(__caller_alloc);
    __builder.build_stacktrace(__skip + 1, __max_depth);
    basic_stacktrace<_Allocator> __ret{__caller_alloc};
    __ret.__entries_.reserve(__builder.__entries_.size());
    for (auto& __base : __builder.__entries_) {
      __ret.__entries_.emplace_back(__base.to_stacktrace_entry());
    }
    return __ret;
  }

  _LIBCPP_EXPORTED_FROM_ABI constexpr ~basic_stacktrace() = default;

  _LIBCPP_EXPORTED_FROM_ABI basic_stacktrace() noexcept(__kNoThrowDflConstruct) : basic_stacktrace(allocator_type()) {}

  _LIBCPP_EXPORTED_FROM_ABI explicit basic_stacktrace(const allocator_type& __alloc) noexcept
      : __alloc_(__alloc), __entries_(__alloc_) {}

  _LIBCPP_EXPORTED_FROM_ABI basic_stacktrace(basic_stacktrace const& __other) = default;

  _LIBCPP_EXPORTED_FROM_ABI basic_stacktrace(basic_stacktrace&& __other) noexcept = default;

  _LIBCPP_EXPORTED_FROM_ABI basic_stacktrace(basic_stacktrace const& __other, allocator_type const& __alloc)
      : __alloc_(__alloc), __entries_(__other.__entries_, __alloc) {}

  _LIBCPP_EXPORTED_FROM_ABI basic_stacktrace(basic_stacktrace&& __other, allocator_type const& __alloc)
      : __alloc_(__alloc) {
    if (__kAlwaysEqual || __alloc_ == __other.__alloc_) {
      __entries_ = std::move(__other.__entries_);
    } else {
      // "moving" from a container with a different allocator; we're forced to copy items instead
      for (auto const& __entry : __other.__entries_) {
        __entries_.push_back(__entry);
      }
    }
  }

  _LIBCPP_EXPORTED_FROM_ABI basic_stacktrace& operator=(const basic_stacktrace& __other) {
    if (this == std::addressof(__other)) {
      return *this;
    }
    if (__kPropOnCopy) {
      __alloc_ = __other.__alloc_;
    }
    __entries_ = {__other.__entries_, __alloc_};
    return *this;
  }

  _LIBCPP_EXPORTED_FROM_ABI basic_stacktrace&
  operator=(basic_stacktrace&& __other) noexcept(__kPropOnMove || __kAlwaysEqual) {
    if (this == std::addressof(__other)) {
      return *this;
    }
    if (__kPropOnMove) {
      __alloc_   = __other.__alloc_;
      __entries_ = std::move(__other.__entries_);
    } else {
      auto __allocs_eq = __kAlwaysEqual || __alloc_ == __other.__alloc_;
      if (__allocs_eq) {
        __entries_ = std::move(__other.__entries_);
      } else {
        // "moving" from a container with a different allocator;
        // we're forced to copy items instead
        for (auto const& __entry : __other.__entries_) {
          __entries_.push_back(__entry);
        }
      }
    }
    return *this;
  }

  // clang-format on

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

  _LIBCPP_EXPORTED_FROM_ABI void swap(basic_stacktrace<_Allocator>& __other) noexcept {
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

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP_BASIC_STACKTRACE
