// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___HIVE_HIVE_BITMAP_H
#define _LIBCPP___HIVE_HIVE_BITMAP_H

#include <__assert>
#include <__bit/countl.h>
#include <__bit/countr.h>
#include <__bit/popcount.h>
#include <__config>
#include <__cstddef/size_t.h>
#include <__memory/construct_at.h>
#include <__memory/pointer_traits.h>
#include <limits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_EXPERIMENTAL_HIVE

_LIBCPP_BEGIN_NAMESPACE_STD

// The only code that reads or writes a block's words, and the only code that writes its summaries.
template <class _Block>
struct __hive_bitmap {
  using __word_type _LIBCPP_NODEBUG = typename _Block::__word_type;

  static constexpr size_t __word_bits = numeric_limits<__word_type>::digits;

private:
  // __n may be __word_bits.
  static constexpr __word_type __low_bits(size_t __n) noexcept {
    return __n >= __word_bits ? ~__word_type(0) : (__word_type(1) << __n) - 1;
  }

  // Index of the highest set bit; __x must not be 0.
  static constexpr size_t __last_bit(__word_type __x) noexcept { return __word_bits - 1 - std::__countl_zero(__x); }

  static constexpr size_t __live_words(size_t __cap) noexcept { return (__cap + __word_bits - 1) / __word_bits; }

  // The bits of word __w that name slots below __cap.
  static constexpr __word_type __valid_mask(size_t __cap, size_t __w) noexcept {
    return __w * __word_bits >= __cap ? 0 : __low_bits(__cap - __w * __word_bits);
  }

public:
  // One more than __live_words when __cap % 64 == 0: the bit at index __cap always exists and is 0.
  static constexpr size_t __words_for(size_t __cap) noexcept { return __cap / __word_bits + 1; }

  // Starts the lifetime of the words: call it on fresh words too. Leaves __size_ to the caller.
  static constexpr void __reset(_Block& __b) noexcept {
    const size_t __cap = __b.__capacity_;
    for (size_t __w = 0; __w != __words_for(__cap); ++__w) {
      std::__construct_at(std::__to_address(__b.__words_ + __w), __word_type(0));
    }

    __b.__live_summary_ = 0;
    __b.__free_summary_ = __low_bits(__live_words(__cap));
  }

  // Bit __i; __i may be __b.__capacity_, which always reads 0.
  [[nodiscard]] static constexpr bool __is_live(const _Block& __b, size_t __i) noexcept {
    _LIBCPP_ASSERT_INTERNAL(__i <= __b.__capacity_, "hive: slot index past the capacity");
    return ((__b.__words_[__i / __word_bits] >> (__i % __word_bits)) & 1) != 0;
  }

  // Index of the lowest free slot; the block must have one. Needs no valid mask: a free-summary bit says its word has
  // a valid zero bit, and the valid bits are the low ones, so the lowest zero bit is valid.
  [[nodiscard]] static constexpr size_t __find_free(const _Block& __b) noexcept {
    _LIBCPP_ASSERT_INTERNAL(__b.__free_summary_ != 0, "hive: searching a full block for a free slot");

    const size_t __w    = std::__countr_zero(__b.__free_summary_);
    const size_t __slot = __w * __word_bits + std::__countr_zero(~__b.__words_[__w]);
    _LIBCPP_ASSERT_INTERNAL(__slot < __b.__capacity_, "hive: free summary names a full word");
    return __slot;
  }

  // Sets bit __i, which must be clear, and the summaries. Leaves __size_ to the caller.
  static constexpr void __mark_live(_Block& __b, size_t __i) noexcept {
    const size_t __cap = __b.__capacity_;
    _LIBCPP_ASSERT_INTERNAL(__i < __cap, "hive: slot index past the capacity");
    _LIBCPP_ASSERT_INTERNAL(!__is_live(__b, __i), "hive: marking a live slot live");
    const size_t __w = __i / __word_bits;

    const __word_type __word = __b.__words_[__w] | (__word_type(1) << (__i % __word_bits));
    __b.__words_[__w]        = __word;
    __b.__live_summary_ |= __word_type(1) << __w;
    // A word that fills loses its free-summary bit. __w is a live word, so its valid bits are the low
    // __cap - __w * __word_bits (all 64 below the last word).
    const bool __filled = __word == __low_bits(__cap - __w * __word_bits);
    __b.__free_summary_ &= ~(__word_type(__filled) << __w);
  }

  // Index of the first live slot at or after __i, or __b.__capacity_ when there is none. __i may be __b.__capacity_,
  // whose bit always exists and is 0, but not more. Trusts __live_summary_.
  [[nodiscard]] static constexpr size_t __next_live(const _Block& __b, size_t __i) noexcept {
    const size_t __cap = __b.__capacity_;
    _LIBCPP_ASSERT_INTERNAL(__i <= __cap, "hive: slot index past the capacity");

    const size_t __w         = __i / __word_bits;
    const __word_type __rest = __b.__words_[__w] >> (__i % __word_bits);
    if (__rest != 0) {
      return __i + std::__countr_zero(__rest);
    }

    const __word_type __later = __b.__live_summary_ & ~__low_bits(__w + 1);
    if (__later == 0) {
      return __cap;
    }

    const size_t __next = std::__countr_zero(__later);
    _LIBCPP_ASSERT_INTERNAL(__b.__words_[__next] != 0, "hive: live summary names an empty word");
    return __next * __word_bits + std::__countr_zero(__b.__words_[__next]);
  }

  // Index of the last live slot at or before __i, or __b.__capacity_ when there is none. __i may be __b.__capacity_,
  // as in __next_live. Trusts __live_summary_.
  [[nodiscard]] static constexpr size_t __prev_live(const _Block& __b, size_t __i) noexcept {
    const size_t __cap = __b.__capacity_;
    _LIBCPP_ASSERT_INTERNAL(__i <= __cap, "hive: slot index past the capacity");

    const size_t __w         = __i / __word_bits;
    const __word_type __rest = __b.__words_[__w] << (__word_bits - 1 - __i % __word_bits);
    if (__rest != 0) {
      return __i - std::__countl_zero(__rest);
    }

    const __word_type __earlier = __b.__live_summary_ & __low_bits(__w);
    if (__earlier == 0) {
      return __cap;
    }

    const size_t __prev = __last_bit(__earlier);
    _LIBCPP_ASSERT_INTERNAL(__b.__words_[__prev] != 0, "hive: live summary names an empty word");
    return __prev * __word_bits + __last_bit(__b.__words_[__prev]);
  }

  // Index of the first live slot, or __b.__capacity_ when the block is empty.
  [[nodiscard]] static constexpr size_t __first_live(const _Block& __b) noexcept { return __next_live(__b, 0); }

  // Index of the last live slot, or __b.__capacity_ when the block is empty.
  [[nodiscard]] static constexpr size_t __last_live(const _Block& __b) noexcept {
    return __prev_live(__b, __b.__capacity_);
  }

  // Cannot check that a live bit means a constructed element.
  [[nodiscard]] static constexpr bool __invariants(const _Block& __b) noexcept {
    const size_t __cap = __b.__capacity_;
    if (__cap == 0 || __cap > _Block::__max_capacity) {
      return false;
    }

    const size_t __live = __live_words(__cap);

    size_t __count             = 0;
    __word_type __live_summary = 0;
    __word_type __free_summary = 0;
    for (size_t __w = 0; __w != __words_for(__cap); ++__w) {
      const __word_type __word       = __b.__words_[__w];
      const __word_type __valid_bits = __valid_mask(__cap, __w); // not __valid: a SAL macro in the Windows SDK
      if ((__word & ~__valid_bits) != 0) {
        return false;
      }
      __count += std::__popcount(__word);
      if (__w < __live) {
        __live_summary |= __word_type(__word != 0) << __w;
        __free_summary |= __word_type((~__word & __valid_bits) != 0) << __w;
      }
    }

    return __b.__size_ == __count && __b.__live_summary_ == __live_summary && __b.__free_summary_ == __free_summary;
  }
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_EXPERIMENTAL_HIVE

_LIBCPP_POP_MACROS

#endif // _LIBCPP___HIVE_HIVE_BITMAP_H
