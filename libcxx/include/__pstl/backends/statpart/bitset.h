//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_BACKENDS_STATPART_BITSET_H
#define _LIBCPP___PSTL_BACKENDS_STATPART_BITSET_H

#include <__assert>
#include <__config>
#include <__cstddef/size_t.h>
#include <__memory/unique_ptr.h>
#include <__new/align_val_t.h>
#include <__new/global_new_delete.h>
#include <__new/interference_size.h>
#include <__new/nothrow_t.h>
#include <__optional/nullopt_t.h>
#include <__optional/optional.h>
#include <climits> // for CHAR_BIT

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS
namespace __pstl {

namespace __statpart {

// __bitset_view is a non-owning view to a dynamically-sized bitset.
// It consists of size_t words, its bit length it's always rounded to the number of bits in a size_t.
struct _LIBCPP_HIDE_FROM_ABI __bitset_view {
  // The number of bits in a single size_t word.
  static constexpr size_t __bits_per_word_ = CHAR_BIT * sizeof(size_t);

  // The pointer to the bitset words.
  size_t* __bits_;

  // The number of of size_t words in the bitset.
  size_t __words_;

  // Checks if the specified bit is true.
  _LIBCPP_HIDE_FROM_ABI bool __test(size_t __pos) const noexcept {
    size_t __word_index = __pos / __bits_per_word_;
    size_t __bit_index  = __pos % __bits_per_word_;
    return (__bits_[__word_index] & (size_t(1) << __bit_index)) != 0;
  }

  // Sets the specified bit to true.
  _LIBCPP_HIDE_FROM_ABI void __set(size_t __pos) noexcept {
    size_t __word_index = __pos / __bits_per_word_;
    size_t __bit_index  = __pos % __bits_per_word_;
    __bits_[__word_index] |= (size_t(1) << __bit_index);
  }

  // Sets the specified bit to false.
  _LIBCPP_HIDE_FROM_ABI void __reset(size_t __pos) noexcept {
    size_t __word_index = __pos / __bits_per_word_;
    size_t __bit_index  = __pos % __bits_per_word_;
    __bits_[__word_index] &= ~(size_t(1) << __bit_index);
  }

  // Sets all bits to false.
  _LIBCPP_HIDE_FROM_ABI void __reset() noexcept {
    for (size_t __i = 0; __i < __words_; ++__i) {
      __bits_[__i] = 0;
    }
  }
};

// __sharded_dynamic_bitset is an owned dynamic bitset that consists of shards of equal length residing in a contiguous
// memory block.
// Upon construction the number of shards and the number of bits per each shard is provided.
// The bit size of each shard is always rounded up to the cache line, such that they can be used exclusively without
// false sharing.
struct _LIBCPP_HIDE_FROM_ABI __sharded_dynamic_bitset {
  struct _LIBCPP_HIDE_FROM_ABI __deleter {
    void _LIBCPP_HIDE_FROM_ABI operator()(void* p) const noexcept {
#  if _LIBCPP_HAS_ALIGNED_ALLOCATION
      ::operator delete[](p, align_val_t{hardware_destructive_interference_size});
#  else
      ::operator delete[](p);
#  endif
    }
  };

  unique_ptr<size_t[], __deleter> __bits_;
  size_t __shards_count_;
  size_t __words_per_shard_;

  _LIBCPP_HIDE_FROM_ABI __sharded_dynamic_bitset(size_t __shards_count, size_t __shard_size) noexcept
      : __shards_count_(__shards_count) {
    static_assert(hardware_destructive_interference_size % sizeof(size_t) == 0,
                  "hardware_destructive_interference_size must be divisible by sizeof(size_t)");
    _LIBCPP_ASSERT_INTERNAL(__shards_count > 0, "number of shards must be positive");
    _LIBCPP_ASSERT_INTERNAL(__shard_size > 0, "shard size must be positive");
    constexpr size_t __bits_per_cacheline = hardware_destructive_interference_size * CHAR_BIT;
    size_t __cachelines_per_shard         = (__shard_size + __bits_per_cacheline - 1) / __bits_per_cacheline;
    __words_per_shard_   = __cachelines_per_shard * (hardware_destructive_interference_size / sizeof(size_t));
    size_t __total_words = __shards_count * __words_per_shard_;
#  if _LIBCPP_HAS_ALIGNED_ALLOCATION
    __bits_.reset(new (align_val_t{hardware_destructive_interference_size}, nothrow) size_t[__total_words]);
#  else
    __bits_.reset(new (nothrow) size_t[__total_words]);
#  endif
  }

  // Produces a view of the specified shard.
  _LIBCPP_HIDE_FROM_ABI __bitset_view __shard(size_t __shard_idx) const noexcept {
    return {__bits_.get() + __shard_idx * __words_per_shard_, __words_per_shard_};
  }
};

// Creates a sharded dynamic bitset with the specified number of shards and capacity of each shard.
// Returns nullopt in the case of allocation failure.
inline _LIBCPP_HIDE_FROM_ABI optional<__sharded_dynamic_bitset>
__make_sharded_dynamic_bitset(size_t __shards_count, size_t __shard_size) noexcept {
  __sharded_dynamic_bitset __bitset{__shards_count, __shard_size};
  if (!__bitset.__bits_)
    return nullopt;
  return __bitset;
}

} // namespace __statpart

} // namespace __pstl
_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_BACKENDS_STATPART_BITSET_H
