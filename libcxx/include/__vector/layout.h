//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___VECTOR_LAYOUT_H
#define _LIBCPP___VECTOR_LAYOUT_H

#include <__assert>
#include <__config>
#include <__memory/allocator_traits.h>
#include <__memory/compressed_pair.h>
#include <__memory/swap_allocator.h>
#include <__type_traits/is_nothrow_constructible.h>
#include <__utility/move.h>
#include <__utility/swap.h>
#include <__split_buffer>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

/// Defines `std::vector`'s storage layout and any operations that are affected by a change in the
/// layout.
///
/// Dynamically-sized arrays like `std::vector` have several different representations. libc++
/// supports two different layouts for `std::vector`:
///
///   * pointer-based layout
///   * size-based layout
//
/// We describe these layouts below. All vector representations have a pointer that points to where
/// the memory is allocated (called `__begin_`).
///
/// **Pointer-based layout**
///
/// The pointer-based layout uses two more pointers in addition to `__begin_`. The second pointer
/// (called `__end_`) past the end of the part of the buffer that holds valid elements. The pointer
/// (called `__capacity_`) points past the end of the allocated buffer. This is the default
/// representation for libc++ due to historical reasons.
///
/// The second pointer has three primary use-cases:
///   * to compute the size of the vector; and
///   * to construct the past-the-end iterator; and
///   * to indicate where the next element should be appended.
///
/// The third pointer is used to compute the capacity of the vector, which lets the vector know how
/// many elements can be added to the vector before a reallocation is necessary.
///
///    __begin_ = 0xE4FD0, __end_ = 0xE4FF0, __capacity_ = 0xE5000
///                 0xE4FD0                             0xE4FF0           0xE5000
///                    ↓                                   ↓                 ↓
///    ┌───────────────┬────────┬────────┬────────┬────────┬────────┬────────┬─────────────────────┐
///    │ ????????????? │   3174 │   5656 │    648 │    489 │ ------ │ ------ │ ??????????????????? │
///    └───────────────┴────────┴────────┴────────┴────────┴────────┴────────┴─────────────────────┘
///                    ↑                                   ↑                 ↑
///                __begin_                             __end_          __capacity_
///
///    Figure 1: A visual representation of a pointer-based `std::vector<short>`. This vector has
///    four elements, with the capacity to store six.
///
/// This is the default layout for libc++.
///
/// **Size-based layout**
///
/// The size-based layout uses integers to track its size and capacity, and computes pointers to
/// past-the-end of the valid range and the whole buffer only when it's necessary. This layout is
/// opt-in, but yields a significant performance boost relative to the pointer-based layout (see
/// below).
///
///    __begin_ = 0xE4FD0, __size_ = 4, __capacity_ = 6
///                 0xE4FD0
///                    ↓
///    ┌───────────────┬────────┬────────┬────────┬────────┬────────┬────────┬─────────────────────┐
///    │ ????????????? │   3174 │   5656 │    648 │    489 │ ------ │ ------ │ ??????????????????? │
///    └───────────────┴────────┴────────┴────────┴────────┴────────┴────────┴─────────────────────┘
///                    ↑
///                __begin_
///
///    Figure 2: A visual representation of this a pointer-based layout. Blank boxes are not a part
///    of the vector's allocated buffer. Boxes with numbers are valid elements within the vector,
///    and boxes with `xx` have been allocated, but aren't being used as elements right now.
//
/// We conducted an extensive A/B test on production software to confirm that the size-based layout
/// improves compute performance by 0.5%, and decreases system memory usage by up to 0.33%.
///
/// **Class design**
///
/// __vector_layout was designed with the following goals:
///    1. to abstractly represent the buffer's boundaries; and
///    2. to limit the number of `#ifdef` blocks that a reader needs to pass through; and
///    3. given (1) and (2), to have no logically identical components in multiple `#ifdef` clauses.
///
/// To facilitate these goals, there is a single `__vector_layout` definition. Users must choose
/// their vector's layout when libc++ is being configured, so there we don't need to manage multiple
/// vector layout types (e.g. `__vector_size_layout`, `__vector_pointer_layout`, etc.). In doing so,
/// we reduce a significant portion of duplicate code.
template <class _Tp, class _Allocator>
class __vector_layout {
public:
  using value_type _LIBCPP_NODEBUG = _Tp;
  using allocator_type _LIBCPP_NODEBUG = _Allocator;
  using __alloc_traits _LIBCPP_NODEBUG = allocator_traits<allocator_type>;
  using size_type _LIBCPP_NODEBUG = typename __alloc_traits::size_type;
  using pointer _LIBCPP_NODEBUG = typename __alloc_traits::pointer;
  using const_pointer _LIBCPP_NODEBUG = typename __alloc_traits::const_pointer;
#ifdef _LIBCPP_ABI_SIZE_BASED_VECTOR
  using _SplitBuffer _LIBCPP_NODEBUG = __split_buffer<_Tp, _Allocator, __split_buffer_size_layout>;
  using __boundary_type _LIBCPP_NODEBUG = size_type;
#else
  using _SplitBuffer _LIBCPP_NODEBUG = __split_buffer<_Tp, _Allocator, __split_buffer_pointer_layout>;
  using __boundary_type _LIBCPP_NODEBUG = pointer;
#endif

  // Cannot be defaulted, since `_LIBCPP_COMPRESSED_PAIR` isn't an aggregate before C++14.
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI
  __vector_layout() _NOEXCEPT_(is_nothrow_default_constructible<allocator_type>::value)
  : __capacity_()
  {}

  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI explicit
  __vector_layout(allocator_type const& __a) _NOEXCEPT_(is_nothrow_copy_constructible<allocator_type>::value)
  : __capacity_()
  , __alloc_(__a)
  {}

  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI
  explicit __vector_layout(allocator_type&& __a) _NOEXCEPT_(is_nothrow_move_constructible<allocator_type>::value)
  : __capacity_()
  , __alloc_(std::move(__a))
  {}

  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI __vector_layout(__vector_layout&& __other) _NOEXCEPT_(is_nothrow_move_constructible<allocator_type>::value)
  : __begin_(std::move(__other.__begin_))
  , __boundary_(std::move(__other.__boundary_))
  , __capacity_(std::move(__other.__capacity_))
  , __alloc_(std::move(__other.__alloc_))
  {
    __other.__begin_ = nullptr;
    __other.__boundary_ = {};
    __other.__capacity_ = {};
  }

  /// Returns a reference to the stored allocator.
  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI
  allocator_type& __alloc() _NOEXCEPT {
    return __alloc_;
  }

  /// Returns a reference to the stored allocator.
  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI
  allocator_type const& __alloc() const _NOEXCEPT {
    return __alloc_;
  }

  /// Returns a pointer to the beginning of the buffer.
  ///
  /// `__begin_ptr()` is not called `data()` because `vector::data()` returns `T*`, but `__begin_`
  /// is allowed to be a fancy pointer.
  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI
  pointer __begin_ptr() const _NOEXCEPT {
    return __begin_;
  }

  /// Returns the value that the layout uses to determine the vector's size.
  ///
  /// `__boundary_representation()` should only be used when directly operating on the layout from
  /// outside `__vector_layout`. Its result must be used with type deduction to avoid compile-time
  /// failures.
  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI
  __boundary_type __boundary_representation() const _NOEXCEPT {
    return __boundary_;
  }

  /// Returns the value that the layout uses to determine the vector's capacity.
  ///
  /// `__capacity_representation()` should only be used when directly operating on the layout from
  /// outside `__vector_layout`. Its result must be used with type deduction to avoid compile-time
  /// failures.
  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI
  __boundary_type __capacity_representation() const _NOEXCEPT {
    return __capacity_;
  }

  /// Returns how many elements can be added before a reallocation occurs.
  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI
  size_type __remaining_capacity() const _NOEXCEPT {
    return __capacity_ - __boundary_;
  }

  /// Determines if a reallocation is necessary.
  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI
  bool __is_full() const _NOEXCEPT {
    return __boundary_ == __capacity_;
  }

  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI
  void __set_valid_range(pointer __begin, size_type __size) _NOEXCEPT {
    __begin_ = __begin;
    __set_boundary(__size);
  }

  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI
  void __set_valid_range(pointer __begin, pointer __end) _NOEXCEPT {
    __begin_ = __begin;
    __set_boundary(__end);
  }

  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI
  void swap(__vector_layout& __other) _NOEXCEPT {
    std::swap(__begin_, __other.__begin_);
    std::swap(__boundary_, __other.__boundary_);
    std::swap(__capacity_, __other.__capacity_);
    std::__swap_allocator(__alloc_, __other.__alloc_);
  }

  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI
  void __swap_layouts(_SplitBuffer& __other) _NOEXCEPT {
    __other.__swap_layouts(__begin_, __boundary_, __capacity_);
  }

  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI
  void __move_layout(__vector_layout& __other) _NOEXCEPT {
    __begin_ = __other.__begin_;
    __boundary_ = __other.__boundary_;
    __capacity_ = __other.__capacity_;

    __other.__begin_ = nullptr;
    __other.__boundary_ = {};
    __other.__capacity_ = {};
  }

  // The following member functions must be implemented per vector layout.
  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI size_type size() const _NOEXCEPT;
  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI size_type capacity() const _NOEXCEPT;
  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI bool empty() const _NOEXCEPT;
  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI _Tp& back() _NOEXCEPT;
  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI _Tp const& back() const _NOEXCEPT;
  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI pointer __end_ptr() const _NOEXCEPT;
  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI bool __invariants() const _NOEXCEPT;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void __set_boundary(size_type __size) _NOEXCEPT;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void __set_boundary(pointer __end) _NOEXCEPT;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void __set_capacity(size_type __size) _NOEXCEPT;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void __set_capacity(pointer __end) _NOEXCEPT;
private:
  pointer __begin_ = nullptr;
  __boundary_type __boundary_{};

#ifdef _LIBCPP_ABI_SIZE_BASED_VECTOR
  size_type __capacity_ = 0;
  [[no_unique_address]] allocator_type __alloc_;
#else
  _LIBCPP_COMPRESSED_PAIR(pointer, __capacity_, allocator_type, __alloc_);
#endif
};

#ifdef _LIBCPP_ABI_SIZE_BASED_VECTOR
template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 auto __vector_layout<_Tp, _Alloc>::size() const _NOEXCEPT -> size_type {
  return __boundary_;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 auto __vector_layout<_Tp, _Alloc>::capacity() const _NOEXCEPT -> size_type {
  return __capacity_;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 auto __vector_layout<_Tp, _Alloc>::empty() const _NOEXCEPT -> bool {
  return __boundary_ == 0;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 auto __vector_layout<_Tp, _Alloc>::back() _NOEXCEPT -> _Tp& {
  return __begin_[__boundary_ - 1];
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 auto __vector_layout<_Tp, _Alloc>::back() const _NOEXCEPT -> _Tp const& {
  return __begin_[__boundary_ - 1];
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 auto __vector_layout<_Tp, _Alloc>::__end_ptr() const _NOEXCEPT -> pointer {
  return __begin_ + __boundary_;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 bool __vector_layout<_Tp, _Alloc>::__invariants() const _NOEXCEPT {
  if (__begin_ == nullptr)
    return __boundary_ == 0 && __capacity_ == 0;
  return !(__boundary_ > __capacity_);
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void __vector_layout<_Tp, _Alloc>::__set_boundary(size_type __n) _NOEXCEPT {
  __boundary_ = __n;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void __vector_layout<_Tp, _Alloc>::__set_boundary(pointer __ptr) _NOEXCEPT {
  __boundary_ = static_cast<size_type>(__ptr - __begin_);
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void __vector_layout<_Tp, _Alloc>::__set_capacity(size_type __n) _NOEXCEPT {
  __capacity_ = __n;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void __vector_layout<_Tp, _Alloc>::__set_capacity(pointer __ptr) _NOEXCEPT {
  __capacity_ = static_cast<size_type>(__ptr - __begin_);
}
#else
template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 auto __vector_layout<_Tp, _Alloc>::size() const _NOEXCEPT -> size_type {
  return static_cast<size_type>(__boundary_ - __begin_);
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 auto __vector_layout<_Tp, _Alloc>::capacity() const _NOEXCEPT -> size_type {
  return static_cast<size_type>(__capacity_ - __begin_);
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 auto __vector_layout<_Tp, _Alloc>::empty() const _NOEXCEPT -> bool {
  return __begin_ == __boundary_;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 auto __vector_layout<_Tp, _Alloc>::back() _NOEXCEPT -> _Tp& {
  return __boundary_[-1];
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 auto __vector_layout<_Tp, _Alloc>::back() const _NOEXCEPT -> _Tp const& {
  return __boundary_[-1];
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 auto __vector_layout<_Tp, _Alloc>::__end_ptr() const _NOEXCEPT -> pointer {
  return __boundary_;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 bool __vector_layout<_Tp, _Alloc>::__invariants() const _NOEXCEPT {
  if (__begin_ == nullptr)
    return __boundary_ == 0 && __capacity_ == 0;
  if (__begin_ > __boundary_)
    return false;
  if (__begin_ == __capacity_)
    return false;
  return !(__boundary_ > __capacity_);
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void __vector_layout<_Tp, _Alloc>::__set_boundary(size_type __n) _NOEXCEPT {
  __boundary_ = __begin_ + __n;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void __vector_layout<_Tp, _Alloc>::__set_boundary(pointer __ptr) _NOEXCEPT {
  __boundary_ = __ptr;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void __vector_layout<_Tp, _Alloc>::__set_capacity(size_type __n) _NOEXCEPT {
  __capacity_ = __begin_ + __n;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void __vector_layout<_Tp, _Alloc>::__set_capacity(pointer __ptr) _NOEXCEPT {
  __capacity_ = __ptr;
}
#endif // _LIBCPP_ABI_SIZE_BASED_VECTOR

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___VECTOR_LAYOUT_H
