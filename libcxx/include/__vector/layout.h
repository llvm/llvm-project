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
#include <__debug_utils/sanitizers.h>
#include <__memory/allocator_traits.h>
#include <__memory/compressed_pair.h>
#include <__memory/pointer_traits.h>
#include <__memory/swap_allocator.h>
#include <__memory/uninitialized_algorithms.h>
#include <__split_buffer>
#include <__type_traits/is_nothrow_constructible.h>
#include <__utility/exchange.h>
#include <__utility/move.h>
#include <__utility/swap.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

/// Defines `std::vector`'s storage layout and any operations that are affected by a change in the
/// layout.
///
/// `std::vector` can be represented in a variety of ways. Each representation strongly influences
/// the codegen when calling vector operations, which can significantly impact runtime performance
/// and memory utilisation. libc++ provides two alternative layouts for `std::vector`, although only
/// one can be active for an enitre binary:
///
///   * pointer-based layout (stable ABI default)
///   * size-based layout (unstable ABI alternative)
//
/// We describe these layouts below. All vector representations have a pointer that points to where
/// the memory is allocated (called `__begin_`).
///
/// **Pointer-based layout**
///
/// The pointer-based layout uses two more pointers in addition to `__begin_`. The second pointer
/// (called `__end_`) points past the end of the part of the buffer that holds valid elements.
/// Another pointer (called `__capacity_`) points past the end of the allocated buffer. The original
/// libc++ `std::vector` implementation only provided the pointer-based layout. libc++ continues to
/// use the pointer-based layout, by default, in order to maintain binary compatibility with
/// existing software.
///
/// The `__end_` pointer has three primary use-cases:
///   * to compute the size of the vector; and
///   * to construct the past-the-end iterator; and
///   * to indicate where the next element should be appended.
///
/// The `__capacity_` is used to compute the capacity of the vector, which lets the vector know how
/// many elements can be added to the vector before a reallocation is necessary.
///
///    __begin_ = 0xE4FD0, __end_ = 0xE4FF0, __capacity_ = 0xE5000
///                 0xE4FD0                             0xE4FF0           0xE5000
///                    v                                   v                 v
///    +---------------+--------+--------+--------+--------+--------+--------+---------------------+
///    | ????????????? |   3174 |   5656 |    648 |    489 | ------ | ------ | ??????????????????? |
///    +---------------+--------+--------+--------+--------+--------+--------+---------------------+
///                    ^                                   ^                 ^
///                __begin_                             __end_          __capacity_
///
///    Figure 1: A visual representation of a pointer-based `std::vector<short>`. This vector has
///    four elements, with the capacity to store six. Boxes with numbers are valid elements within
///    the vector, and boxes with `xx` have been allocated, but aren't being used as elements right
///    now.
///
/// This is the default layout for libc++.
///
/// **Size-based layout**
///
/// The size-based layout uses integers to track its size and capacity, and computes pointers to
/// past-the-end of the valid range and the whole buffer only when it's necessary. Programs using
/// the size-based layout have been measured to yield improved compute and memory performance over
/// the pointer-based layout. Despite these promising measurements, the size-based layout is opt-in,
/// to preserve ABI compatibility with prebuilt binaries. Given the improved performance, we
/// recommend preferring the size-based layout in the absence of such ABI constraints.
///
///    __begin_ = 0xE4FD0, __size_ = 4, __capacity_ = 6
///                 0xE4FD0
///                    v
///    +---------------+--------+--------+--------+--------+--------+--------+---------------------+
///    | ????????????? |   3174 |   5656 |    648 |    489 | ------ | ------ | ??????????????????? |
///    +---------------+--------+--------+--------+--------+--------+--------+---------------------+
///                    ^
///                __begin_
///
///    Figure 2: A visual representation of this a size-based layout. Blank boxes are not a part
///    of the vector's allocated buffer.
///
/// **Class design**
///
/// __vector_layout was designed with the following goals:
///    1. to abstractly represent the buffer's boundaries; and
///    2. to limit the number of `#ifdef` blocks that a reader needs to pass through; and
///    3. given (1) and (2), to have no logically identical components in multiple `#ifdef` clauses.
///
/// To facilitate these goals, there is a single `__vector_layout` definition. Users must choose
/// their vector's layout when libc++ is being configured, so there is no need to manage multiple
/// vector layout types (e.g. `__vector_size_layout`, `__vector_pointer_layout`, etc.). In doing so,
/// we reduce a significant portion of duplicate code.
template <class _Tp, class _Allocator>
class __vector_layout {
public:
  using value_type _LIBCPP_NODEBUG     = _Tp;
  using allocator_type _LIBCPP_NODEBUG = _Allocator;
  using __alloc_traits _LIBCPP_NODEBUG = allocator_traits<allocator_type>;
  using size_type _LIBCPP_NODEBUG      = typename __alloc_traits::size_type;
  using pointer _LIBCPP_NODEBUG        = typename __alloc_traits::pointer;
  using const_pointer _LIBCPP_NODEBUG  = typename __alloc_traits::const_pointer;
#ifdef _LIBCPP_ABI_SIZE_BASED_VECTOR
  using _SplitBuffer _LIBCPP_NODEBUG = __split_buffer<_Tp, _Allocator, __split_buffer_size_layout>;
  using __bound_type _LIBCPP_NODEBUG = size_type;
#else
  using _SplitBuffer _LIBCPP_NODEBUG = __split_buffer<_Tp, _Allocator, __split_buffer_pointer_layout>;
  using __bound_type _LIBCPP_NODEBUG = pointer;
#endif

  // Cannot be defaulted, since `_LIBCPP_COMPRESSED_PAIR` isn't an aggregate before C++14.
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI __vector_layout()
      _NOEXCEPT_(is_nothrow_default_constructible<allocator_type>::value)
      : __capacity_() {}

  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI explicit __vector_layout(allocator_type const& __a)
      _NOEXCEPT_(is_nothrow_copy_constructible<allocator_type>::value)
      : __capacity_(), __alloc_(__a) {}

  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI explicit __vector_layout(allocator_type&& __a)
      _NOEXCEPT_(is_nothrow_move_constructible<allocator_type>::value)
      : __capacity_(), __alloc_(std::move(__a)) {}

  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI __vector_layout(__vector_layout&& __other)
      _NOEXCEPT_(is_nothrow_move_constructible<allocator_type>::value);

  /// Returns a reference to the stored allocator.
  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI allocator_type& __alloc() _NOEXCEPT {
    return __alloc_;
  }

  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI allocator_type const&
  __alloc() const _NOEXCEPT {
    return __alloc_;
  }

  /// Returns a pointer to the beginning of the buffer.
  ///
  /// `__begin_ptr()` is not called `data()` because `vector::data()` returns `T*`, but `__begin_`
  /// is allowed to be a fancy pointer.
  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI pointer __begin_ptr() _NOEXCEPT {
    return __begin_;
  }

  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI const_pointer __begin_ptr() const _NOEXCEPT {
    return __begin_;
  }

  /// Returns a built-in pointer to the beginning of the buffer.
  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI _Tp* __data() _NOEXCEPT {
    return std::__to_address(__begin_);
  }

  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI _Tp const* __data() const _NOEXCEPT {
    return std::__to_address(__begin_);
  }

  /// Returns the value that the layout uses to determine the vector's size.
  ///
  /// `__bound_representation()` should only be used when directly operating on the layout from
  /// outside `__vector_layout`. Its result must be used with type deduction to avoid compile-time
  /// failures.
  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI __bound_type
  __bound_representation() const _NOEXCEPT;

  /// Returns how many elements can be added before a reallocation occurs.
  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI size_type
  __remaining_capacity() const _NOEXCEPT;

  /// Determines if a reallocation is necessary.
  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI bool __is_full() const _NOEXCEPT;

  /// Sets the member pointing to the first element in the vector to `__new_begin`, the member used
  /// to obtain the vector's bound to the equivalent of `__new_size`, and the member that represents
  /// the vector's capacity to the equivalent to `__new_capacity`.
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void
  __set_layout(pointer __new_begin, size_type __new_size, size_type __new_capacity) _NOEXCEPT;

  /// Sets the member used to obtain the vector's bound to the equivalent of `__ptr`.
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void __set_bound_using_pointer(pointer __ptr) _NOEXCEPT;

  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void __reset_without_allocator() _NOEXCEPT;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void __swap(__vector_layout& __other) _NOEXCEPT;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void __swap_layouts(_SplitBuffer& __other) _NOEXCEPT;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void
  __move_assign_without_allocator(__vector_layout& __other) _NOEXCEPT;

  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void __relocate(_SplitBuffer& __buffer);
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI pointer
  __relocate_with_pivot(_SplitBuffer& __buffer, pointer __pivot);

  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI size_type __size() const _NOEXCEPT;
  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI size_type __capacity() const _NOEXCEPT;
  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI bool __empty() const _NOEXCEPT;
  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI _Tp& __back() _NOEXCEPT;
  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI _Tp const& __back() const _NOEXCEPT;
  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI pointer __end_ptr() _NOEXCEPT;
  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI const_pointer __end_ptr() const _NOEXCEPT;
  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI pointer __capacity_ptr() _NOEXCEPT;
  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI const_pointer __capacity_ptr() const _NOEXCEPT;
  [[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI bool __invariants() const _NOEXCEPT;

private:
  pointer __begin_ = nullptr;

#ifdef _LIBCPP_ABI_SIZE_BASED_VECTOR
  size_type __size_     = 0;
  size_type __capacity_ = 0;
  [[no_unique_address]] allocator_type __alloc_;
#else
  pointer __end_ = nullptr;
  _LIBCPP_COMPRESSED_PAIR(pointer, __capacity_ = nullptr, allocator_type, __alloc_);
#endif

  _LIBCPP_CONSTEXPR_SINCE_CXX20
  _LIBCPP_HIDE_FROM_ABI void __annotate_contiguous_container(const void* __old_mid, const void* __new_mid) const {
    std::__annotate_contiguous_container<_Allocator>(__data(), __data() + __capacity(), __old_mid, __new_mid);
  }

  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void __annotate_new(size_type __current_size) const _NOEXCEPT {
    __annotate_contiguous_container(__data() + __capacity(), __data() + __current_size);
  }

  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void __annotate_delete() const _NOEXCEPT {
    __annotate_contiguous_container(__data() + __size(), __data() + __capacity());
  }
};

#ifdef _LIBCPP_ABI_SIZE_BASED_VECTOR
template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 __vector_layout<_Tp, _Alloc>::__vector_layout(__vector_layout&& __other)
    _NOEXCEPT_(is_nothrow_move_constructible<allocator_type>::value)
    : __begin_(std::__exchange(__other.__begin_, nullptr)),
      __size_(std::__exchange(__other.__size_, 0)),
      __capacity_(std::__exchange(__other.__capacity_, 0)),
      __alloc_(std::move(__other.__alloc_)) {}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 typename __vector_layout<_Tp, _Alloc>::__bound_type
__vector_layout<_Tp, _Alloc>::__bound_representation() const _NOEXCEPT {
  return __size_;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 typename __vector_layout<_Tp, _Alloc>::size_type
__vector_layout<_Tp, _Alloc>::__remaining_capacity() const _NOEXCEPT {
  return __capacity_ - __size_;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 bool __vector_layout<_Tp, _Alloc>::__is_full() const _NOEXCEPT {
  return __size_ == __capacity_;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void __vector_layout<_Tp, _Alloc>::__set_layout(
    pointer __new_begin, size_type __new_size, size_type __new_capacity) _NOEXCEPT {
  __begin_    = __new_begin;
  __size_     = __new_size;
  __capacity_ = __new_capacity;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void __vector_layout<_Tp, _Alloc>::__set_bound_using_pointer(pointer __ptr) _NOEXCEPT {
  __size_ = static_cast<size_type>(__ptr - __begin_);
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void __vector_layout<_Tp, _Alloc>::__reset_without_allocator() _NOEXCEPT {
  __begin_    = nullptr;
  __size_     = 0;
  __capacity_ = 0;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void __vector_layout<_Tp, _Alloc>::__swap(__vector_layout& __other) _NOEXCEPT {
  using std::swap;
  swap(__begin_, __other.__begin_);
  swap(__size_, __other.__size_);
  swap(__capacity_, __other.__capacity_);
  std::__swap_allocator(__alloc_, __other.__alloc_);
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void __vector_layout<_Tp, _Alloc>::__swap_layouts(_SplitBuffer& __other) _NOEXCEPT {
  __other.__swap_layouts(__begin_, __size_, __capacity_);
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void
__vector_layout<_Tp, _Alloc>::__move_assign_without_allocator(__vector_layout& __other) _NOEXCEPT {
  __begin_    = __other.__begin_;
  __size_     = __other.__size_;
  __capacity_ = __other.__capacity_;

  __other.__reset_without_allocator();
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void __vector_layout<_Tp, _Alloc>::__relocate(_SplitBuffer& __buffer) {
  __annotate_delete();
  __buffer.__relocate(__begin_, __size_, __capacity_);
  __annotate_new(__size_);
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 typename __vector_layout<_Tp, _Alloc>::pointer
__vector_layout<_Tp, _Alloc>::__relocate_with_pivot(_SplitBuffer& __buffer, pointer __pivot) {
  __annotate_delete();
  auto __result = __buffer.__relocate_with_pivot(__pivot, __begin_, __size_, __capacity_);
  __annotate_new(__size_);
  return __result;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 typename __vector_layout<_Tp, _Alloc>::size_type
__vector_layout<_Tp, _Alloc>::__size() const _NOEXCEPT {
  return __size_;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 typename __vector_layout<_Tp, _Alloc>::size_type
__vector_layout<_Tp, _Alloc>::__capacity() const _NOEXCEPT {
  return __capacity_;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 bool __vector_layout<_Tp, _Alloc>::__empty() const _NOEXCEPT {
  return __size_ == 0;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 _Tp& __vector_layout<_Tp, _Alloc>::__back() _NOEXCEPT {
  return __begin_[__size_ - 1];
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 _Tp const& __vector_layout<_Tp, _Alloc>::__back() const _NOEXCEPT {
  return __begin_[__size_ - 1];
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 typename __vector_layout<_Tp, _Alloc>::pointer
__vector_layout<_Tp, _Alloc>::__end_ptr() _NOEXCEPT {
  return __begin_ + __size_;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 typename __vector_layout<_Tp, _Alloc>::const_pointer
__vector_layout<_Tp, _Alloc>::__end_ptr() const _NOEXCEPT {
  return __begin_ + __size_;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 typename __vector_layout<_Tp, _Alloc>::pointer
__vector_layout<_Tp, _Alloc>::__capacity_ptr() _NOEXCEPT {
  return __begin_ + __size_;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 typename __vector_layout<_Tp, _Alloc>::const_pointer
__vector_layout<_Tp, _Alloc>::__capacity_ptr() const _NOEXCEPT {
  return __begin_ + __capacity_;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 bool __vector_layout<_Tp, _Alloc>::__invariants() const _NOEXCEPT {
  if (__begin_ == nullptr)
    return __size_ == 0 && __capacity_ == 0;
  return __size_ <= __capacity_;
}
#else
template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 __vector_layout<_Tp, _Alloc>::__vector_layout(__vector_layout&& __other)
    _NOEXCEPT_(is_nothrow_move_constructible<allocator_type>::value)
    : __begin_(std::__exchange(__other.__begin_, nullptr)),
      __end_(std::__exchange(__other.__end_, nullptr)),
      __capacity_(std::__exchange(__other.__capacity_, nullptr)),
      __alloc_(std::move(__other.__alloc_)) {}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 typename __vector_layout<_Tp, _Alloc>::__bound_type
__vector_layout<_Tp, _Alloc>::__bound_representation() const _NOEXCEPT {
  return __end_;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 typename __vector_layout<_Tp, _Alloc>::size_type
__vector_layout<_Tp, _Alloc>::__remaining_capacity() const _NOEXCEPT {
  return __capacity_ - __end_;
}

template <class _Tp, class _Alloc>
[[__nodiscard__]] _LIBCPP_CONSTEXPR_SINCE_CXX20 bool __vector_layout<_Tp, _Alloc>::__is_full() const _NOEXCEPT {
  return __end_ == __capacity_;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void __vector_layout<_Tp, _Alloc>::__set_layout(
    pointer __new_begin, size_type __new_size, size_type __new_capacity) _NOEXCEPT {
  __begin_    = __new_begin;
  __end_      = __new_begin + __new_size;
  __capacity_ = __new_begin + __new_capacity;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void __vector_layout<_Tp, _Alloc>::__set_bound_using_pointer(pointer __ptr) _NOEXCEPT {
  __end_ = __ptr;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void __vector_layout<_Tp, _Alloc>::__reset_without_allocator() _NOEXCEPT {
  __begin_    = nullptr;
  __end_      = nullptr;
  __capacity_ = nullptr;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void __vector_layout<_Tp, _Alloc>::__swap(__vector_layout& __other) _NOEXCEPT {
  using std::swap;
  swap(__begin_, __other.__begin_);
  swap(__end_, __other.__end_);
  swap(__capacity_, __other.__capacity_);
  std::__swap_allocator(__alloc_, __other.__alloc_);
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void __vector_layout<_Tp, _Alloc>::__swap_layouts(_SplitBuffer& __other) _NOEXCEPT {
  __other.__swap_layouts(__begin_, __end_, __capacity_);
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void
__vector_layout<_Tp, _Alloc>::__move_assign_without_allocator(__vector_layout& __other) _NOEXCEPT {
  __begin_    = __other.__begin_;
  __end_      = __other.__end_;
  __capacity_ = __other.__capacity_;

  __other.__reset_without_allocator();
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void __vector_layout<_Tp, _Alloc>::__relocate(_SplitBuffer& __buffer) {
  __annotate_delete();
  __buffer.__relocate(__begin_, __end_, __capacity_);
  __annotate_new(__size());
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 typename __vector_layout<_Tp, _Alloc>::pointer
__vector_layout<_Tp, _Alloc>::__relocate_with_pivot(_SplitBuffer& __buffer, pointer __pivot) {
  __annotate_delete();
  auto __result = __buffer.__relocate_with_pivot(__pivot, __begin_, __end_, __capacity_);
  __annotate_new(__size());
  return __result;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 typename __vector_layout<_Tp, _Alloc>::size_type
__vector_layout<_Tp, _Alloc>::__size() const _NOEXCEPT {
  return static_cast<size_type>(__end_ - __begin_);
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 typename __vector_layout<_Tp, _Alloc>::size_type
__vector_layout<_Tp, _Alloc>::__capacity() const _NOEXCEPT {
  return static_cast<size_type>(__capacity_ - __begin_);
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 bool __vector_layout<_Tp, _Alloc>::__empty() const _NOEXCEPT {
  return __begin_ == __end_;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 _Tp& __vector_layout<_Tp, _Alloc>::__back() _NOEXCEPT {
  return __end_[-1];
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 _Tp const& __vector_layout<_Tp, _Alloc>::__back() const _NOEXCEPT {
  return __end_[-1];
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 typename __vector_layout<_Tp, _Alloc>::pointer
__vector_layout<_Tp, _Alloc>::__end_ptr() _NOEXCEPT {
  return __end_;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 typename __vector_layout<_Tp, _Alloc>::const_pointer
__vector_layout<_Tp, _Alloc>::__end_ptr() const _NOEXCEPT {
  return __end_;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 typename __vector_layout<_Tp, _Alloc>::pointer
__vector_layout<_Tp, _Alloc>::__capacity_ptr() _NOEXCEPT {
  return __capacity_;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 typename __vector_layout<_Tp, _Alloc>::const_pointer
__vector_layout<_Tp, _Alloc>::__capacity_ptr() const _NOEXCEPT {
  return __capacity_;
}

template <class _Tp, class _Alloc>
_LIBCPP_CONSTEXPR_SINCE_CXX20 bool __vector_layout<_Tp, _Alloc>::__invariants() const _NOEXCEPT {
  if (__begin_ == nullptr)
    return __end_ == nullptr && __capacity_ == nullptr;
  if (__begin_ > __end_)
    return false;
  if (__begin_ == __capacity_)
    return false;
  return __end_ <= __capacity_;
}
#endif // _LIBCPP_ABI_SIZE_BASED_VECTOR

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___VECTOR_LAYOUT_H
