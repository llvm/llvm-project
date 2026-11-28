// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___HIVE_HIVE_STORAGE_H
#define _LIBCPP___HIVE_HIVE_STORAGE_H

#include <__algorithm/clamp.h>
#include <__algorithm/min.h>
#include <__assert>
#include <__config>
#include <__cstddef/size_t.h>
#include <__hive/hive_bitmap.h>
#include <__hive/hive_block.h>
#include <__hive/hive_block_capacity.h>
#include <__hive/hive_iterator.h>
#include <__hive/hive_limits.h>
#include <__memory/addressof.h>
#include <__memory/allocation_guard.h>
#include <__memory/allocator_traits.h>
#include <__memory/construct_at.h>
#include <__memory/pointer_traits.h>
#include <__memory/uninitialized_algorithms.h>
#include <__type_traits/is_trivially_destructible.h>
#include <__utility/exchange.h>
#include <__utility/forward.h>
#include <__utility/pair.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_EXPERIMENTAL_HIVE

_LIBCPP_BEGIN_NAMESPACE_STD

// A block's words and slots are one allocation of these. The slots start a whole number of units in, which is what
// makes casting a unit pointer to a slot pointer valid for fancy pointers.
template <size_t _Align>
struct alignas(_Align) __hive_unit {
  unsigned char __bytes_[_Align];
};

template <class _Tp, class _Allocator>
class __hive_storage {
private:
  using __alloc_traits _LIBCPP_NODEBUG = allocator_traits<_Allocator>;
  using __void_ptr _LIBCPP_NODEBUG     = typename __alloc_traits::void_pointer;
  using __block_type _LIBCPP_NODEBUG =
      __hive_block<_Tp, __void_ptr>; // not __block: a keyword in Objective-C++ and under -fblocks
  using __block_pointer _LIBCPP_NODEBUG  = typename __block_type::__block_pointer;
  using __index_type _LIBCPP_NODEBUG     = typename __block_type::__index_type;
  using __block_capacity _LIBCPP_NODEBUG = __hive_block_capacity<_Tp, __void_ptr, typename __alloc_traits::size_type>;
  using __bitmap _LIBCPP_NODEBUG         = __hive_bitmap<__block_type>;
  using __word_pointer _LIBCPP_NODEBUG   = typename __block_type::__word_pointer;
  using __slot_pointer _LIBCPP_NODEBUG   = typename __block_type::__slot_pointer;

  static constexpr size_t __align = __block_capacity::__align;

  using __unit _LIBCPP_NODEBUG         = __hive_unit<__align>;
  using __block_alloc _LIBCPP_NODEBUG  = __rebind_alloc<__alloc_traits, __block_type>;
  using __unit_alloc _LIBCPP_NODEBUG   = __rebind_alloc<__alloc_traits, __unit>;
  using __unit_pointer _LIBCPP_NODEBUG = typename allocator_traits<__unit_alloc>::pointer;

public:
  using allocator_type _LIBCPP_NODEBUG = _Allocator;
  using size_type _LIBCPP_NODEBUG      = typename __alloc_traits::size_type;
  using iterator _LIBCPP_NODEBUG       = __hive_iterator<_Tp, __void_ptr, false>;
  using const_iterator _LIBCPP_NODEBUG = __hive_iterator<_Tp, __void_ptr, true>;

  constexpr explicit __hive_storage(const allocator_type& __a) noexcept : __alloc_(__a) {}

  constexpr __hive_storage(hive_limits __l, const allocator_type& __a) noexcept : __alloc_(__a) { __set_limits(__l); }

  constexpr ~__hive_storage() {
    __destroy_blocks(__first_);
    __destroy_blocks(__reserved_head_);
  }

  __hive_storage(const __hive_storage&)            = delete;
  __hive_storage& operator=(const __hive_storage&) = delete;

  [[nodiscard]] constexpr size_type __size() const noexcept { return __size_; }
  [[nodiscard]] constexpr size_type __capacity() const noexcept { return __capacity_; }
  [[nodiscard]] constexpr hive_limits __limits() const noexcept { return hive_limits(__min_, __max_); }

  [[nodiscard]] constexpr allocator_type& __alloc() noexcept { return __alloc_; }
  [[nodiscard]] constexpr const allocator_type& __alloc() const noexcept { return __alloc_; }

  // The first live slot of the first block, or the empty hive's end.
  [[nodiscard]] constexpr iterator __begin() noexcept {
    return __first_ != nullptr ? std::__make_hive_iterator<iterator>(__first_, __bitmap::__first_live(*__first_))
                               : iterator();
  }
  [[nodiscard]] constexpr const_iterator __begin() const noexcept {
    return __first_ != nullptr ? std::__make_hive_iterator<const_iterator>(__first_, __bitmap::__first_live(*__first_))
                               : const_iterator();
  }

  // One past the last block's slots, so that ++ reaches it with no capacity compare; {nullptr, 0} when empty.
  [[nodiscard]] constexpr iterator __end() noexcept {
    return __last_ != nullptr ? std::__make_hive_iterator<iterator>(__last_, __last_->__capacity_) : iterator();
  }
  [[nodiscard]] constexpr const_iterator __end() const noexcept {
    return __last_ != nullptr ? std::__make_hive_iterator<const_iterator>(__last_, __last_->__capacity_)
                              : const_iterator();
  }

  // Capacity of the next new block: the size, within the limits and rounded up to whole words, but never above
  // __max_, so limits such as {3, 3} and {65, 100} stay exact.
  [[nodiscard]] constexpr size_t __next_capacity() const noexcept {
    const size_t __wanted = std::clamp<size_t>(__size_, __min_, __max_);
    const size_t __whole  = (__wanted + __bitmap::__word_bits - 1) / __bitmap::__word_bits * __bitmap::__word_bits;
    return std::min<size_t>(__whole, __max_);
  }

  // Constructs an element in a free slot and returns its block and index. A throw from the allocator or the
  // constructor leaves the storage unchanged: nothing is marked, counted or linked before the construction, and a
  // new block is freed by its handle.
  template <class... _Args>
  constexpr pair<__block_pointer, size_t> __emplace(_Args&&... __args) {
    if (__free_head_ != nullptr) {
      const __block_pointer __b = __free_head_;
      const size_t __i          = __bitmap::__find_free(*__b);
      __alloc_traits::construct(__alloc_, std::__to_address(__b->__slots_ + __i), std::forward<_Args>(__args)...);

      __commit(*__b, __i);
      if (__b->__free_summary_ == 0) {
        __pop_free();
      }
      return {__b, __i};
    }

    return __emplace_in_empty_block(std::forward<_Args>(__args)...);
  }

  // The reserved and new-block cases of __emplace. Kept out of line: inlined, the block handle's frame and stack
  // protector land on the free-list path. An empty block gives slot 0.
  template <class... _Args>
  _LIBCPP_NOINLINE constexpr pair<__block_pointer, size_t> __emplace_in_empty_block(_Args&&... __args) {
    if (__reserved_head_ != nullptr) {
      const __block_pointer __b = __reserved_head_;
      __alloc_traits::construct(__alloc_, std::__to_address(__b->__slots_), std::forward<_Args>(__args)...);

      __reserved_head_ = __b->__next_;
      __commit(*__b, 0);
      __link_back(__b);
      return {__b, 0};
    }

    __block_handle __h        = __allocate_block(__next_capacity());
    const __block_pointer __b = __h.__get();
    __alloc_traits::construct(__alloc_, std::__to_address(__b->__slots_), std::forward<_Args>(__args)...);

    (void)__h.__release_ptr();
    __capacity_ += __b->__capacity_;
    __commit(*__b, 0);
    __link_back(__b);
    return {__b, 0};
  }

  // Destroys every element and retires every active block to the reserved list. __capacity_ is unchanged.
  constexpr void __clear() noexcept {
    for (__block_pointer __b = __first_; __b != nullptr;) {
      const __block_pointer __next = __b->__next_;
      __destroy_elements(*__b);
      __bitmap::__reset(*__b);
      __b->__next_     = __reserved_head_;
      __reserved_head_ = __b;
      __b              = __next;
    }

    __first_     = nullptr;
    __last_      = nullptr;
    __free_head_ = nullptr;
    __size_      = 0;
  }

  // Walks every block and word: for tests only. Allocates nothing, and every walk is bounded, so a corrupt list
  // returns false instead of looping.
  [[nodiscard]] constexpr bool __invariants() const noexcept {
    if (!__block_capacity::__within_hard_limits(__limits())) {
      return false;
    }
    if ((__first_ == nullptr) != (__last_ == nullptr)) {
      return false;
    }
    if (__first_ != nullptr && (__first_->__prev_ != nullptr || __last_->__next_ != nullptr)) {
      return false;
    }

    // Strictly increasing numbers also rule out a cycle.
    size_type __size      = 0;
    size_type __capacity  = 0;
    size_type __with_free = 0;
    for (__block_pointer __b = __first_; __b != nullptr; __b = __b->__next_) {
      if (!__bitmap::__invariants(*__b) || __b->__size_ == 0 || __b->__capacity_ < __min_ ||
          __b->__capacity_ > __max_) {
        return false;
      }
      if (__b->__next_ == nullptr ? __b != __last_
                                  : __b->__next_->__prev_ != __b || __b->__next_->__number_ <= __b->__number_) {
        return false;
      }
      __size += __b->__size_;
      __capacity += __b->__capacity_;
      __with_free += __b->__free_summary_ != 0;
    }

    // Active blocks with a free slot. Equal counts plus consistent back links stand in for membership.
    size_type __in_free_list = 0;
    for (__block_pointer __b = __free_head_, __prev = nullptr; __b != nullptr; __prev = __b, __b = __b->__next_free_) {
      if (__b->__prev_free_ != __prev || __b->__size_ == 0 || __b->__free_summary_ == 0 ||
          ++__in_free_list > __with_free) {
        return false;
      }
    }
    if (__in_free_list != __with_free) {
      return false;
    }

    // Every block has capacity >= 1, so the bound on __capacity also stops a cycle.
    for (__block_pointer __b = __reserved_head_; __b != nullptr; __b = __b->__next_) {
      if (!__bitmap::__invariants(*__b) || __b->__size_ != 0) {
        return false;
      }
      __capacity += __b->__capacity_;
      if (__capacity > __capacity_) {
        return false;
      }
    }

    return __size == __size_ && __capacity == __capacity_;
  }

private:
  __block_pointer __first_         = nullptr;
  __block_pointer __last_          = nullptr;
  __block_pointer __free_head_     = nullptr; // active blocks with free slots
  size_type __size_                = 0;
  size_type __capacity_            = 0;
  __block_pointer __reserved_head_ = nullptr;
  __index_type __min_              = __block_capacity::__default_min;
  __index_type __max_              = __block_capacity::__default_max;
  _LIBCPP_NO_UNIQUE_ADDRESS allocator_type __alloc_;

  constexpr void __set_limits(hive_limits __l) noexcept {
    __l    = __block_capacity::__clamp(__l);
    __min_ = static_cast<__index_type>(__l.min);
    __max_ = static_cast<__index_type>(__l.max);
  }

  // Owns a block that is in no list and not counted in __capacity_; frees it unless released.
  class __block_handle {
  public:
    constexpr __block_handle(__hive_storage& __s, __block_pointer __b) noexcept
        : __s_(std::addressof(__s)), __b_(__b) {}
    constexpr __block_handle(__block_handle&& __other) noexcept : __s_(__other.__s_), __b_(__other.__release_ptr()) {}
    constexpr ~__block_handle() {
      if (__b_ != nullptr) {
        __s_->__deallocate_block(__b_);
      }
    }

    [[nodiscard]] constexpr __block_pointer __get() const noexcept { return __b_; }
    [[nodiscard]] constexpr __block_pointer __release_ptr() noexcept { return std::exchange(__b_, nullptr); }

  private:
    __hive_storage* __s_;
    __block_pointer __b_;
  };

  // Slots first, so a failed header allocation frees them. The unit count fits the allocator's size_type because
  // __cap is within the hard limits.
  [[nodiscard]] constexpr __block_handle __allocate_block(size_t __cap) {
    _LIBCPP_ASSERT_INTERNAL(__block_capacity::__hard_min <= __cap && __cap <= __block_capacity::__hard_max,
                            "hive: block capacity outside the hard limits");

    __allocation_guard<__unit_alloc> __slots(__alloc_, __block_capacity::__units_for(__cap));
    __allocation_guard<__block_alloc> __header(__alloc_, 1);

    const __unit_pointer __p = __slots.__get();
    std::__construct_at(
        std::addressof(*__header.__get()),
        static_cast<__word_pointer>(static_cast<__void_ptr>(__p)),
        static_cast<__slot_pointer>(static_cast<__void_ptr>(__p + __block_capacity::__word_units(__cap))),
        static_cast<__index_type>(__cap));
    __bitmap::__reset(*__header.__get());

    (void)__slots.__release_ptr();
    return __block_handle(*this, __header.__release_ptr());
  }

  // Records a constructed element: its bit, the summaries, both sizes.
  constexpr void __commit(__block_type& __b, size_t __i) noexcept {
    __bitmap::__mark_live(__b, __i);
    ++__b.__size_;
    ++__size_;
  }

  // Links a block that is in no list at the back of the active list, and on the free list unless it is full.
  constexpr void __link_back(__block_pointer __b) noexcept {
    __b->__prev_ = __last_;
    __b->__next_ = nullptr;
    if (__last_ != nullptr) {
      __b->__number_   = __last_->__number_ + 1;
      __last_->__next_ = __b;
    } else {
      __b->__number_ = 0;
      __first_       = __b;
    }
    __last_ = __b;

    if (__b->__free_summary_ != 0) {
      __b->__prev_free_ = nullptr;
      __b->__next_free_ = __free_head_;
      if (__free_head_ != nullptr) {
        __free_head_->__prev_free_ = __b;
      }
      __free_head_ = __b;
    }
  }

  // Takes the head off the free list: the block __emplace just filled.
  constexpr void __pop_free() noexcept {
    __free_head_ = __free_head_->__next_free_;
    if (__free_head_ != nullptr) {
      __free_head_->__prev_free_ = nullptr;
    }
  }

  // Leaves the words and summaries stale.
  constexpr void __destroy_elements(__block_type& __b) noexcept {
    if constexpr (!is_trivially_destructible_v<_Tp> || !__allocator_has_trivial_destroy_v<_Allocator, _Tp>) {
      for (size_t __i = __bitmap::__next_live(__b, 0); __i != __b.__capacity_;
           __i        = __bitmap::__next_live(__b, __i + 1)) {
        __alloc_traits::destroy(__alloc_, std::__to_address(__b.__slots_ + __i));
      }
    }
    __b.__size_ = 0;
  }

  // Destroys the elements and frees every block from __b on. Leaves the heads and counters stale.
  constexpr void __destroy_blocks(__block_pointer __b) noexcept {
    while (__b != nullptr) {
      const __block_pointer __next = __b->__next_;
      __destroy_elements(*__b);
      __deallocate_block(__b);
      __b = __next;
    }
  }

  // Leaves the lists and __capacity_ to the caller.
  constexpr void __deallocate_block(__block_pointer __b) noexcept {
    _LIBCPP_ASSERT_INTERNAL(__b->__size_ == 0, "hive: freeing a block with live elements");

    // Read from the header before it is destroyed.
    const size_t __units     = __block_capacity::__units_for(__b->__capacity_);
    const __unit_pointer __p = static_cast<__unit_pointer>(static_cast<__void_ptr>(__b->__words_));

    std::__destroy_at(std::addressof(*__b));
    __block_alloc __header_alloc(__alloc_);
    allocator_traits<__block_alloc>::deallocate(__header_alloc, __b, 1);

    __unit_alloc __slot_alloc(__alloc_);
    allocator_traits<__unit_alloc>::deallocate(__slot_alloc, __p, __units);
  }
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_EXPERIMENTAL_HIVE

_LIBCPP_POP_MACROS

#endif // _LIBCPP___HIVE_HIVE_STORAGE_H
