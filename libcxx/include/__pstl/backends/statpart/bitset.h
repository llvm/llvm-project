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
class _LIBCPP_HIDE_FROM_ABI __sharded_dynamic_bitset {
public:
  // Produces a view of the specified shard.
  _LIBCPP_HIDE_FROM_ABI __bitset_view __shard(size_t __shard_idx) const noexcept {
    _LIBCPP_ASSERT_INTERNAL(__shard_idx < __shards_count_, "__shard_idx must be within bounds");
    // NB! Reaching for size_ts beyond the bounds of the underlying __aligned_cache_line is technically an UB,
    // but this is mostly theoretical.
    return {reinterpret_cast<size_t*>(__bits_.get()) + __shard_idx * __words_per_shard_, __words_per_shard_};
  }

  // Returns the number of bitset shards in the bitset.
  _LIBCPP_HIDE_FROM_ABI size_t __shards_count() const noexcept { return __shards_count_; }

  // Returns the number of words per shard in the bitset.
  _LIBCPP_HIDE_FROM_ABI size_t __words_per_shard() const noexcept { return __words_per_shard_; }

private:
  // Internally __sharded_dynamic_bitset allocates a dynamic array of __aligned_cache_line,
  // instead of allocating raw "size_t[]" and using an explicit "align_val_t" in the new expression.
  // This approach meshes better with TMO rewrites and conditional availability of aligned allocation.
  struct alignas(hardware_destructive_interference_size) __aligned_cache_line {
    static constexpr size_t __bits_per_cacheline_  = hardware_destructive_interference_size * CHAR_BIT;
    static constexpr size_t __words_per_cacheline_ = hardware_destructive_interference_size / sizeof(size_t);
    size_t __words_[__words_per_cacheline_];
  };

  _LIBCPP_HIDE_FROM_ABI __sharded_dynamic_bitset(size_t __shards_count, size_t __shard_size) noexcept
      : __shards_count_(__shards_count) {
    static_assert(hardware_destructive_interference_size % sizeof(size_t) == 0,
                  "hardware_destructive_interference_size must be divisible by sizeof(size_t)");
    _LIBCPP_ASSERT_INTERNAL(__shards_count > 0, "number of shards must be positive");
    _LIBCPP_ASSERT_INTERNAL(__shard_size > 0, "shard size must be positive");
    size_t __cachelines_per_shard =
        (__shard_size + __aligned_cache_line::__bits_per_cacheline_ - 1) / __aligned_cache_line::__bits_per_cacheline_;
    size_t __total_cachelines = __shards_count * __cachelines_per_shard;
    __bits_.reset(new (nothrow) __aligned_cache_line[__total_cachelines]);
    __words_per_shard_ = __cachelines_per_shard * __aligned_cache_line::__words_per_cacheline_;
  }

  unique_ptr<__aligned_cache_line[]> __bits_;
  size_t __shards_count_;
  size_t __words_per_shard_;

  friend _LIBCPP_HIDE_FROM_ABI optional<__sharded_dynamic_bitset>
  __make_sharded_dynamic_bitset(size_t, size_t) noexcept;
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
