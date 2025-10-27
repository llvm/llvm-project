// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_BASIC_STACKTRACE_H
#define _LIBCPP_BASIC_STACKTRACE_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#include <__assert>
#include <__cstddef/size_t.h>
#include <__functional/function.h>
#include <__functional/hash.h>
#include <__iterator/iterator.h>
#include <__iterator/reverse_iterator.h>
#include <__memory/allocator_traits.h>
#include <__memory_resource/polymorphic_allocator.h>
#include <__new/allocate.h>
#include <__type_traits/is_nothrow_constructible.h>
#include <__vector/vector.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#if _LIBCPP_HAS_LOCALIZATION
#  include <__fwd/format.h>
#  include <__fwd/ostream.h>
#endif // _LIBCPP_HAS_LOCALIZATION

#if !defined(_WIN32)
#  include <unwind.h>
#endif

#if _LIBCPP_STD_VER >= 23 && _LIBCPP_AVAILABILITY_HAS_STACKTRACE

#  include <__stacktrace/stacktrace_entry.h>

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

  _Str_Alloc<char> __string_alloc_;

  using _EntryIters _LIBCPP_NODEBUG = _Iters<stacktrace_entry, _Entry>;
  function<_EntryIters()> __entry_iters_;
  function<_Entry&()> __entry_append_;

  template <class _Allocator>
  _LIBCPP_HIDE_FROM_ABI
  _Trace(_Allocator const& __alloc, function<_EntryIters()> __entry_iters, function<_Entry&()> __entry_append)
      : __string_alloc_(std::move(_Str_Alloc<char>::make(__alloc))),
        __entry_iters_(__entry_iters),
        __entry_append_(__entry_append) {}

#  ifndef _WIN32
  // Use <unwind.h> or <libunwind.h> to collect addresses in the stack.
  // Defined in this header so it can be reliably force-inlined.
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_ALWAYS_INLINE void unwind_addrs(size_t __skip, size_t __depth);
#  endif

  // On Windows, there are DLLs which take care of all this (dbghelp, psapi), with
  // zero overlap with any other OS, so it's in its own file, impl_windows.cpp.
  // For all other platforms we implement these "find" methods.

  // Resolve instruction addresses to their respective images, dealing with possible ASLR.
  _LIBCPP_EXPORTED_FROM_ABI void find_images();

  // Resolve addresses to symbols if possible.
  _LIBCPP_EXPORTED_FROM_ABI void find_symbols();

  // Resolve addresses to source locations if possible.
  _LIBCPP_EXPORTED_FROM_ABI void find_source_locs();

  _LIBCPP_HIDE_FROM_ABI void finish_stacktrace() {
    find_images();
    find_symbols();
    find_source_locs();
  }

  _LIBCPP_EXPORTED_FROM_ABI ostream& write_to(ostream& __os) const;
  _LIBCPP_EXPORTED_FROM_ABI string to_string() const;

  _LIBCPP_HIDE_FROM_ABI _Str __create_str() { return _Str(__string_alloc_); }
};

} // namespace __stacktrace

// (19.6.4)
// Class template basic_stacktrace [stacktrace.basic]

class stacktrace_entry;

template <class _Allocator>
class basic_stacktrace : private __stacktrace::_Trace {
  friend struct hash<basic_stacktrace<_Allocator>>;
  friend struct __stacktrace::_Trace;

  using _ATraits _LIBCPP_NODEBUG            = allocator_traits<_Allocator>;
  constexpr static bool __kPropOnCopyAssign = _ATraits::propagate_on_container_copy_assignment::value;
  constexpr static bool __kPropOnMoveAssign = _ATraits::propagate_on_container_move_assignment::value;
  constexpr static bool __kPropOnSwap       = _ATraits::propagate_on_container_swap::value;
  constexpr static bool __kAlwaysEqual      = _ATraits::is_always_equal::value;
  constexpr static bool __kNoThrowAlloc     = noexcept(noexcept(_Allocator().allocate(1)));

  _LIBCPP_NO_UNIQUE_ADDRESS _Allocator __alloc_;

  vector<stacktrace_entry, _Allocator> __entries_;
  _LIBCPP_HIDE_FROM_ABI _EntryIters entry_iters() { return {__entries_.data(), __entries_.size()}; }
  _LIBCPP_HIDE_FROM_ABI __stacktrace::_Entry& entry_append() {
    return (__stacktrace::_Entry&)__entries_.emplace_back();
  }

  _LIBCPP_HIDE_FROM_ABI auto entry_iters_fn() {
    return [this] -> _EntryIters { return entry_iters(); };
  }
  _LIBCPP_HIDE_FROM_ABI auto entry_append_fn() {
    return [this] -> __stacktrace::_Entry& { return entry_append(); };
  }

public:
  // (19.6.4.1)
  // Overview [stacktrace.basic.overview]

  using value_type             = stacktrace_entry;
  using const_reference        = value_type const&;
  using reference              = value_type&;
  using difference_type        = ptrdiff_t;
  using size_type              = size_t;
  using allocator_type         = _Allocator;
  using iterator               = decltype(__entries_.begin());
  using const_iterator         = decltype(__entries_.cbegin());
  using reverse_iterator       = decltype(__entries_.rbegin());
  using const_reverse_iterator = decltype(__entries_.crbegin());

  // (19.6.4.2)
  // Creation and assignment [stacktrace.basic.cons]

  _LIBCPP_ALWAYS_INLINE static basic_stacktrace
  current(const allocator_type& __caller_alloc = allocator_type()) noexcept(__kNoThrowAlloc) {
    size_type __skip      = 0;
    size_type __max_depth = __default_max_depth;
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        __skip <= __skip + __max_depth, "sum of skip and max_depth overflows size_type");
    basic_stacktrace __ret{__caller_alloc};
    __ret.unwind_addrs(__skip, __max_depth);
    __ret.finish_stacktrace();
    return __ret;
  }

  _LIBCPP_ALWAYS_INLINE static basic_stacktrace
  current(size_type __skip, const allocator_type& __caller_alloc = allocator_type()) noexcept(__kNoThrowAlloc) {
    size_type __max_depth = __default_max_depth;
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        __skip <= __skip + __max_depth, "sum of skip and max_depth overflows size_type");
    basic_stacktrace __ret{__caller_alloc};
    __ret.unwind_addrs(__skip, __max_depth);
    __ret.finish_stacktrace();
    return __ret;
  }

  _LIBCPP_ALWAYS_INLINE static basic_stacktrace
  current(size_type __skip,
          size_type __max_depth,
          const allocator_type& __caller_alloc = allocator_type()) noexcept(__kNoThrowAlloc) {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
        __skip <= __skip + __max_depth, "sum of skip and max_depth overflows size_type");
    basic_stacktrace __ret{__caller_alloc};
    if (__max_depth) [[likely]] {
      __ret.unwind_addrs(__skip, __max_depth);
      __ret.finish_stacktrace();
    }
    return __ret;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr ~basic_stacktrace() = default;

  static_assert(sizeof(__stacktrace::_Entry) == sizeof(stacktrace_entry));

  _LIBCPP_HIDE_FROM_ABI explicit basic_stacktrace(const allocator_type& __alloc)
      : _Trace(__alloc, entry_iters_fn(), entry_append_fn()), __alloc_(__alloc), __entries_(__alloc_) {}

  _LIBCPP_HIDE_FROM_ABI basic_stacktrace(basic_stacktrace const& __other, allocator_type const& __alloc)
      : _Trace(__alloc, entry_iters_fn(), entry_append_fn()), __alloc_(__alloc), __entries_(__other.__entries_) {}

  _LIBCPP_HIDE_FROM_ABI basic_stacktrace(basic_stacktrace&& __other, allocator_type const& __alloc)
      : _Trace(__alloc, entry_iters_fn(), entry_append_fn()),
        __alloc_(__alloc),
        __entries_(std::move(__other.__entries_)) {}

  _LIBCPP_HIDE_FROM_ABI basic_stacktrace() noexcept(is_nothrow_default_constructible_v<allocator_type>)
      : basic_stacktrace(allocator_type()) {}

  _LIBCPP_HIDE_FROM_ABI basic_stacktrace(basic_stacktrace const& __other) noexcept
      : basic_stacktrace(__other, _ATraits::select_on_container_copy_construction(__other.__alloc_)) {}

  _LIBCPP_HIDE_FROM_ABI basic_stacktrace(basic_stacktrace&& __other) noexcept
      : basic_stacktrace(std::move(__other), __other.__alloc_) {}

  _LIBCPP_HIDE_FROM_ABI basic_stacktrace& operator=(const basic_stacktrace& __other) {
    if (std::addressof(__other) != this) {
      if (__kPropOnCopyAssign) {
        new (this) basic_stacktrace(__other, __other.__alloc_);
      } else {
        new (this) basic_stacktrace(__other);
      }
    }
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI basic_stacktrace&
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

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI allocator_type get_allocator() const noexcept { return __alloc_; }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI const_iterator begin() const noexcept { return __entries_.begin(); }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI const_iterator end() const noexcept { return __entries_.end(); }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI const_reverse_iterator rbegin() const noexcept { return __entries_.rbegin(); }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI const_reverse_iterator rend() const noexcept { return __entries_.rend(); }

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
_LIBCPP_HIDE_FROM_ABI inline void
swap(basic_stacktrace<_Allocator>& __a, basic_stacktrace<_Allocator>& __b) noexcept(noexcept(__a.swap(__b))) {
  __a.swap(__b);
}

#  if _LIBCPP_HAS_LOCALIZATION
template <class _Allocator>
_LIBCPP_HIDE_FROM_ABI inline ostream& operator<<(ostream& __os, const basic_stacktrace<_Allocator>& __stacktrace) {
  return ((__stacktrace::_Trace const&)__stacktrace).write_to(__os);
}
template <class _Allocator>
_LIBCPP_HIDE_FROM_ABI inline string to_string(const basic_stacktrace<_Allocator>& __stacktrace) {
  return ((__stacktrace::_Trace const&)__stacktrace).to_string();
}
#  endif // _LIBCPP_HAS_LOCALIZATION

// (19.6.6)
// Hash support [stacktrace.basic.hash]

template <class _Allocator>
struct hash<basic_stacktrace<_Allocator>> {
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI size_t operator()(basic_stacktrace<_Allocator> const& __context) const noexcept {
    size_t __ret = 1;
    for (auto const& __entry : __context.__entries_) {
      __ret += hash<uintptr_t>()(__entry.native_handle());
    }
    return __ret;
  }
};

namespace __stacktrace {

#  if defined(_WIN32)

_LIBCPP_EXPORTED_FROM_ABI void _Trace::windows_impl(size_t skip, size_t max_depth)

#  else

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

_LIBCPP_HIDE_FROM_ABI _LIBCPP_ALWAYS_INLINE inline void _Trace::unwind_addrs(size_t __skip, size_t __depth) {
  if (!__depth) {
    return;
  }
  _Unwind_Wrapper __bt{*this, __skip, __depth};
  _Unwind_Backtrace(_Unwind_Wrapper::callback, &__bt);
}

#  endif // _WIN32

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23 && _LIBCPP_AVAILABILITY_HAS_STACKTRACE

_LIBCPP_POP_MACROS

#endif // _LIBCPP_BASIC_STACKTRACE_H
