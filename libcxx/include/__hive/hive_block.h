// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___HIVE_HIVE_BLOCK_H
#define _LIBCPP___HIVE_HIVE_BLOCK_H

#include <__config>
#include <__cstddef/size_t.h>
#include <__memory/pointer_traits.h>
#include <cstdint>
#include <limits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_EXPERIMENTAL_HIVE

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp, class _VoidPtr>
struct __hive_block {
  using __block_pointer _LIBCPP_NODEBUG = __rebind_pointer_t<_VoidPtr, __hive_block>;
  using __slot_pointer _LIBCPP_NODEBUG  = __rebind_pointer_t<_VoidPtr, _Tp>;
  using __index_type _LIBCPP_NODEBUG    = uint16_t;
  using __word_type _LIBCPP_NODEBUG     = uint64_t;
  using __word_pointer _LIBCPP_NODEBUG  = __rebind_pointer_t<_VoidPtr, __word_type>;

  // One summary word over 64 live words.
  static constexpr size_t __max_capacity = 64 * 64;
  static_assert(__max_capacity <= numeric_limits<__index_type>::max());

  constexpr __hive_block(__word_pointer __words, __slot_pointer __slots, __index_type __cap) noexcept
      : __words_(__words), __slots_(__slots), __capacity_(__cap) {}

  // ++, * and an emplace that does not fill the block touch only the fields up to __free_summary_ (the first 48 bytes
  // with raw pointers), so keep them first.
  __block_pointer __next_ = nullptr;
  __word_pointer __words_;
  __slot_pointer __slots_;
  __index_type __capacity_;
  __index_type __size_        = 0;
  __index_type __hint_        = 0;
  __word_type __live_summary_ = 0;
  __word_type __free_summary_ = 0;

  __block_pointer __prev_      = nullptr;
  __block_pointer __next_free_ = nullptr; // list of active blocks with free slots
  __block_pointer __prev_free_ = nullptr;
  size_t __number_             = 0;
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_EXPERIMENTAL_HIVE

_LIBCPP_POP_MACROS

#endif // _LIBCPP___HIVE_HIVE_BLOCK_H
