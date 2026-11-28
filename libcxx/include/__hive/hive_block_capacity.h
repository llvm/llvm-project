// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___HIVE_HIVE_BLOCK_CAPACITY_H
#define _LIBCPP___HIVE_HIVE_BLOCK_CAPACITY_H

#include <__algorithm/clamp.h>
#include <__algorithm/max.h>
#include <__algorithm/min.h>
#include <__config>
#include <__cstddef/size_t.h>
#include <__hive/hive_bitmap.h>
#include <__hive/hive_block.h>
#include <__hive/hive_limits.h>
#include <limits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_EXPERIMENTAL_HIVE

_LIBCPP_BEGIN_NAMESPACE_STD

// _SizeType is the allocator's size_type.
template <class _Tp, class _VoidPtr, class _SizeType>
struct __hive_block_capacity {
  using __block_type _LIBCPP_NODEBUG =
      __hive_block<_Tp, _VoidPtr>; // not __block: a keyword in Objective-C++ and under -fblocks
  using __word_type _LIBCPP_NODEBUG = typename __block_type::__word_type;

  // A block's words and slots are one allocation of units of this size.
  static constexpr size_t __align = std::max(alignof(_Tp), alignof(__word_type));

  static constexpr size_t __units(size_t __bytes) noexcept { return (__bytes + __align - 1) / __align; }

  // The size of the words in units, which is where the slots start.
  static constexpr size_t __word_units(size_t __cap) noexcept {
    return __units(__hive_bitmap<__block_type>::__words_for(__cap) * sizeof(__word_type));
  }

  // __cap * sizeof(_Tp) must not wrap: true for every capacity that __fits.
  static constexpr size_t __units_for(size_t __cap) noexcept {
    return __word_units(__cap) + __units(__cap * sizeof(_Tp));
  }

  // The words of the largest block plus the padding after the words and after the slots.
  static constexpr size_t __max_overhead =
      __hive_bitmap<__block_type>::__words_for(__block_type::__max_capacity) * sizeof(__word_type) + 2 * __align;

  // The most bytes one allocation can be asked for: what size_t and the allocator's size_type can both count.
  // [hive.overview]/5.3 bounds the maximum hard limit by allocator_traits::max_size(), which takes an allocator object
  // that a static limit does not have; this is the part of that bound the type alone gives.
  static constexpr size_t __max_bytes =
      numeric_limits<_SizeType>::max() < numeric_limits<size_t>::max()
          ? static_cast<size_t>(numeric_limits<_SizeType>::max())
          : numeric_limits<size_t>::max();

  // Whether a block of __cap slots can be allocated. The first test keeps __units_for from wrapping.
  static constexpr bool __fits(size_t __cap) noexcept {
    return __cap <= (numeric_limits<size_t>::max() - __max_overhead) / sizeof(_Tp) &&
           __units_for(__cap) <= __max_bytes / __align;
  }

  // The largest capacity that fits, or 0 when none does. __fits is monotonic, so this bisects.
  static constexpr size_t __largest_capacity() noexcept {
    size_t __low  = 0;
    size_t __high = __block_type::__max_capacity;
    while (__low != __high) {
      const size_t __mid = __low + (__high - __low + 1) / 2;
      if (__fits(__mid)) {
        __low = __mid;
      } else {
        __high = __mid - 1;
      }
    }

    return __low;
  }

  static constexpr size_t __hard_min = 1;
  static constexpr size_t __hard_max = __largest_capacity();

  static constexpr size_t __default_max_bytes = 128 * 1024;

  static constexpr size_t __default_min = std::min<size_t>(64, __hard_max);
  static constexpr size_t __default_max = std::clamp(__default_max_bytes / sizeof(_Tp), __default_min, __hard_max);

  static constexpr hive_limits __hard_limits{__hard_min, __hard_max};
  static constexpr hive_limits __default_limits{__default_min, __default_max};

  static constexpr bool __within_hard_limits(hive_limits __l) noexcept {
    return __hard_min <= __l.min && __l.min <= __l.max && __l.max <= __hard_max;
  }

  // Maps any input into the hard limits with min <= max.
  static constexpr hive_limits __clamp(hive_limits __l) noexcept {
    size_t __max = std::clamp(__l.max, __hard_min, __hard_max);
    return hive_limits(std::clamp(__l.min, __hard_min, __max), __max);
  }
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_EXPERIMENTAL_HIVE

_LIBCPP_POP_MACROS

#endif // _LIBCPP___HIVE_HIVE_BLOCK_CAPACITY_H
