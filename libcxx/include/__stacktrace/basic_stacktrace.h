// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___BASIC_STACKTRACE_H
#define _LIBCPP___BASIC_STACKTRACE_H

#include <__assert>
#include <__config>
#include <__cstddef/size_t.h>
#include <__functional/function.h>
#include <__functional/hash.h>
#include <__fwd/format.h>
#include <__iterator/iterator.h>
#include <__iterator/reverse_iterator.h>
#include <__memory/allocator_traits.h>
#include <__memory_resource/polymorphic_allocator.h>
#include <__new/allocate.h>
#include <__stacktrace/stacktrace_entry.h>
#include <__type_traits/is_nothrow_constructible.h>
#include <__vector/vector.h>
#include <cstddef>
#include <iostream>
#include <string>
#include <utility>
#if _LIBCPP_HAS_LOCALIZATION
#  include <__fwd/ostream.h>
#endif // _LIBCPP_HAS_LOCALIZATION
#if !defined(_WIN32)
#  include <unwind.h>
#endif

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 23 && _LIBCPP_AVAILABILITY_HAS_STACKTRACE

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __stacktrace {

template <typename _Tp, typename _Bp = _Tp>
struct _Iters {
  _Tp* __data_{};
  size_t __size_{};

  _Bp* data() { return (_Bp*)__data_; }
  size_t size() const { return __size_; }
  _Bp* begin() { return data(); }
  _Bp* end() { return data() + size(); }
};

struct _Trace {
  constexpr static size_t __default_max_depth  = 64;
  constexpr static size_t __absolute_max_depth = 256;

  using _EntryIters _LIBCPP_NODEBUG = _Iters<stacktrace_entry, _Entry>;
  function<_EntryIters()> __entry_iters_;
  function<_Entry&()> __entry_append_;

  _LIBCPP_HIDE_FROM_ABI _Trace(function<_EntryIters()> __entry_iters, function<_Entry&()> __entry_append)
      : __entry_iters_(__entry_iters), __entry_append_(__entry_append) {}

  _LIBCPP_EXPORTED_FROM_ABI ostream& write_to(ostream& __os) const;
  _LIBCPP_EXPORTED_FROM_ABI string to_string() const;

  _LIBCPP_EXPORTED_FROM_ABI size_t hash() const;
  _LIBCPP_HIDE_FROM_ABI static _Trace& base(auto& __trace);
  _LIBCPP_HIDE_FROM_ABI static _Trace const& base(auto const& __trace);

#  ifdef _WIN32
  // Windows impl uses dbghelp and psapi DLLs to do the full stacktrace operation.
  _LIBCPP_EXPORTED_FROM_ABI void windows_impl(size_t skip, size_t max_depth);
#  else
  // Non-windows: impl separated out into several smaller platform-dependent parts.
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_ALWAYS_INLINE void populate_addrs(size_t __skip, size_t __depth);
  _LIBCPP_EXPORTED_FROM_ABI void populate_images();
#  endif
};

} // namespace __stacktrace

// (19.6.4)
// Class template basic_stacktrace [stacktrace.basic]

class stacktrace_entry;

template <class _Allocator>
class basic_stacktrace : private __stacktrace::_Trace {
  friend struct __stacktrace::_Trace;

  vector<stacktrace_entry, _Allocator> __entries_;

  _LIBCPP_HIDE_FROM_ABI _EntryIters __entry_iters() { return {__entries_.data(), __entries_.size()}; }

  _LIBCPP_HIDE_FROM_ABI __stacktrace::_Entry& __entry_append() {
    return (__stacktrace::_Entry&)__entries_.emplace_back();
  }

  _LIBCPP_HIDE_FROM_ABI auto __entry_iters_fn() {
    return [this] -> _EntryIters { return __entry_iters(); };
  }
  _LIBCPP_HIDE_FROM_ABI auto __entry_append_fn() {
    return [this] -> __stacktrace::_Entry& { return __entry_append(); };
  }

public:
  // (19.6.4.1)
  // Overview [stacktrace.basic.overview]

  using value_type             = stacktrace_entry;
  using const_reference        = const value_type&;
  using reference              = value_type&;
  using const_iterator         = decltype(__entries_.cbegin());
  using iterator               = const_iterator;
  using reverse_iterator       = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using difference_type        = ptrdiff_t;
  using size_type              = size_t;
  using allocator_type         = _Allocator;

  // (19.6.4.2)
  // Creation and assignment [stacktrace.basic.cons]

  _LIBCPP_ALWAYS_INLINE // Omit this function from the trace
  static basic_stacktrace current(const allocator_type& __alloc = allocator_type()) noexcept {
    return current(0, __default_max_depth, __alloc);
  }

  _LIBCPP_ALWAYS_INLINE // Omit this function from the trace
  static basic_stacktrace current(size_type __skip, const allocator_type& __alloc = allocator_type()) noexcept {
    return current(__skip, __default_max_depth, __alloc);
  }

  _LIBCPP_ALWAYS_INLINE // Omit this function from the trace
  static basic_stacktrace
  current(size_type __skip, size_type __max_depth, const allocator_type& __alloc = allocator_type()) noexcept {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        __skip <= __skip + __max_depth, "sum of skip and max_depth overflows size_type");
    basic_stacktrace __ret{__alloc};

    if (__max_depth) {
#  if defined(_WIN32)
      __ret.windows_impl(__skip, __max_depth);
#  else
      __ret.populate_addrs(__skip, __max_depth);
      __ret.populate_images();
#  endif
    }

    return __ret;
  }

  _LIBCPP_HIDE_FROM_ABI basic_stacktrace() noexcept(is_nothrow_default_constructible_v<allocator_type>)
      : basic_stacktrace(allocator_type()) {}

  _LIBCPP_HIDE_FROM_ABI explicit basic_stacktrace(const allocator_type& __alloc) noexcept
      : _Trace(__entry_iters_fn(), __entry_append_fn()), __entries_(__alloc) {}

  _LIBCPP_HIDE_FROM_ABI basic_stacktrace(const basic_stacktrace& __other)
      : _Trace(__entry_iters_fn(), __entry_append_fn()) {
    __entries_ = __other.__entries_;
  }

  _LIBCPP_HIDE_FROM_ABI basic_stacktrace(basic_stacktrace&& __other) noexcept
      : _Trace(__entry_iters_fn(), __entry_append_fn()) {
    __entries_ = std::move(__other.__entries_);
  }

  _LIBCPP_HIDE_FROM_ABI basic_stacktrace(const basic_stacktrace& __other, const allocator_type& __alloc)
      : _Trace(__entry_iters_fn(), __entry_append_fn()), __entries_(__other.__entries_, __alloc) {}

  _LIBCPP_HIDE_FROM_ABI basic_stacktrace(basic_stacktrace&& __other, const allocator_type& __alloc)
      : _Trace(__entry_iters_fn(), __entry_append_fn()), __entries_(std::move(__other.__entries_), __alloc) {}

  _LIBCPP_HIDE_FROM_ABI basic_stacktrace& operator=(const basic_stacktrace& __other) {
    __entries_ = __other.__entries_;
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI basic_stacktrace& operator=(basic_stacktrace&& __other) noexcept(
      allocator_traits<_Allocator>::propagate_on_container_move_assignment::value ||
      allocator_traits<_Allocator>::is_always_equal::value) {
    __entries_ = std::move(__other.__entries_);
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI ~basic_stacktrace() = default;

  // (19.6.4.3)
  // [stacktrace.basic.obs], observers

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI allocator_type get_allocator() const noexcept { return __entries_.get_allocator(); }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI const_iterator begin() const noexcept { return __entries_.begin(); }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI const_iterator end() const noexcept { return __entries_.end(); }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI const_reverse_iterator rbegin() const noexcept { return __entries_.rbegin(); }
 [[nodiscard]]  _LIBCPP_HIDE_FROM_ABI const_reverse_iterator rend() const noexcept { return __entries_.rend(); }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI const_iterator cbegin() const noexcept { return __entries_.cbegin(); }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI const_iterator cend() const noexcept { return __entries_.cend(); }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI const_reverse_iterator crbegin() const noexcept { return __entries_.crbegin(); }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI const_reverse_iterator crend() const noexcept { return __entries_.crend(); }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI bool empty() const noexcept { return __entries_.empty(); }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI size_type size() const noexcept { return __entries_.size(); }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI size_type max_size() const noexcept { return __entries_.max_size(); }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI const_reference operator[](size_type __i) const { return __entries_[__i]; }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI const_reference at(size_type __i) const { return __entries_.at(__i); }

  // (19.6.4.4)
  // [stacktrace.basic.cmp], comparisons

  template <class _Allocator2>
  _LIBCPP_HIDE_FROM_ABI friend bool
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
  _LIBCPP_HIDE_FROM_ABI friend strong_ordering
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

  _LIBCPP_HIDE_FROM_ABI void swap(basic_stacktrace& __other) noexcept(
      allocator_traits<_Allocator>::propagate_on_container_swap::value ||
      allocator_traits<_Allocator>::is_always_equal::value) {
    std::swap(__entries_, __other.__entries_);
  }
};

using stacktrace = basic_stacktrace<allocator<stacktrace_entry>>;

namespace pmr {
using stacktrace = basic_stacktrace<polymorphic_allocator<stacktrace_entry>>;
} // namespace pmr

// (19.6.4.6)
// Non-member functions [stacktrace.basic.nonmem]

template <class _Allocator>
_LIBCPP_HIDE_FROM_ABI void
swap(basic_stacktrace<_Allocator>& __a, basic_stacktrace<_Allocator>& __b) noexcept(noexcept(__a.swap(__b))) {
  __a.swap(__b);
}

#  if _LIBCPP_HAS_LOCALIZATION

template <class _Allocator>
_LIBCPP_HIDE_FROM_ABI string to_string(const basic_stacktrace<_Allocator>& __stacktrace) {
  return ((__stacktrace::_Trace const&)__stacktrace).to_string();
}

template <class _Allocator>
_LIBCPP_HIDE_FROM_ABI ostream& operator<<(ostream& __os, const basic_stacktrace<_Allocator>& __stacktrace) {
  return ((__stacktrace::_Trace const&)__stacktrace).write_to(__os);
}

#  endif // _LIBCPP_HAS_LOCALIZATION

// (19.6.6)
// Hash support [stacktrace.basic.hash]

template <class _Allocator>
struct hash<basic_stacktrace<_Allocator>> {
  _LIBCPP_HIDE_FROM_ABI size_t operator()(basic_stacktrace<_Allocator> const& __trace) const noexcept {
    return __stacktrace::_Trace::base(__trace).hash();
  }
};

namespace __stacktrace {

#  if !defined(_WIN32)

struct _Unwind_Wrapper {
  _Trace& base_;
  size_t skip_;
  size_t maxDepth_;

  _LIBCPP_HIDE_FROM_ABI _Unwind_Reason_Code callback(_Unwind_Context* __ucx) {
    if (skip_) {
      --skip_;
      return _Unwind_Reason_Code::_URC_NO_REASON;
    }
    if (!maxDepth_) {
      return _Unwind_Reason_Code::_URC_NORMAL_STOP;
    }
    --maxDepth_;
    int __ip_before{0};
    auto __ip = _Unwind_GetIPInfo(__ucx, &__ip_before);
    if (!__ip) {
      return _Unwind_Reason_Code::_URC_NORMAL_STOP;
    }
    auto& __entry = base_.__entry_append_();
    auto& __eb    = (_Entry&)__entry;
    __eb.__addr_  = (__ip_before ? __ip : __ip - 1);
    return _Unwind_Reason_Code::_URC_NO_REASON;
  }

  _LIBCPP_HIDE_FROM_ABI static _Unwind_Reason_Code callback(_Unwind_Context* __cx, void* __self) {
    return ((_Unwind_Wrapper*)__self)->callback(__cx);
  }
};

_LIBCPP_HIDE_FROM_ABI _LIBCPP_ALWAYS_INLINE inline void _Trace::populate_addrs(size_t __skip, size_t __depth) {
  if (!__depth) {
    return;
  }
  _Unwind_Wrapper __bt{*this, __skip, __depth};
  _Unwind_Backtrace(_Unwind_Wrapper::callback, &__bt);
}

#  endif // _WIN32

_Trace& _Trace::base(auto& __trace) { return *static_cast<_Trace*>(std::addressof(__trace)); }

_Trace const& _Trace::base(auto const& __trace) { return *static_cast<_Trace const*>(std::addressof(__trace)); }

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23 && _LIBCPP_AVAILABILITY_HAS_STACKTRACE

_LIBCPP_POP_MACROS

#endif // _LIBCPP___BASIC_STACKTRACE_H
