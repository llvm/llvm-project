//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUBRANGE_TEST_TYPES_H
#define SUBRANGE_TEST_TYPES_H

#include "test_macros.h"
#include "test_iterators.h"

int globalBuff[8];

struct Empty {};

using InputIter = cpp17_input_iterator<int*>;
using ForwardIter = forward_iterator<int*>;
using BidirIter = bidirectional_iterator<int*>;

using ForwardSubrange = std::ranges::subrange<ForwardIter, ForwardIter, std::ranges::subrange_kind::unsized>;
using SizedIntPtrSubrange = std::ranges::subrange<int*, int*, std::ranges::subrange_kind::sized>;

struct MoveOnlyForwardIter {
    typedef std::forward_iterator_tag       iterator_category;
    typedef int                             value_type;
    typedef std::ptrdiff_t                  difference_type;
    typedef int*                            pointer;
    typedef int&                            reference;
    typedef MoveOnlyForwardIter             self;

    int *base = nullptr;

    MoveOnlyForwardIter() = default;
    MoveOnlyForwardIter(MoveOnlyForwardIter &&) = default;
    MoveOnlyForwardIter &operator=(MoveOnlyForwardIter&&) = default;
    MoveOnlyForwardIter(MoveOnlyForwardIter const&) = delete;
    constexpr MoveOnlyForwardIter(int *ptr) : base(ptr) { }

    friend bool operator==(const self&, const self&);
    friend constexpr bool operator==(const self& lhs, int* rhs) { return lhs.base == rhs; }

    reference operator*() const;
    pointer operator->() const;
    self& operator++();
    self operator++(int);
    self& operator--();
    self operator--(int);

    constexpr operator pointer() const { return base; }
};

struct SizedSentinelForwardIter {
    typedef std::forward_iterator_tag            iterator_category;
    typedef int                                  value_type;
    typedef std::ptrdiff_t                       difference_type;
    typedef int*                                 pointer;
    typedef int&                                 reference;
    typedef std::make_unsigned_t<std::ptrdiff_t> udifference_type;
    typedef SizedSentinelForwardIter             self;

    int *base;

    SizedSentinelForwardIter() = default;
    constexpr SizedSentinelForwardIter(int *ptr) : base(ptr) { }

    friend constexpr bool operator==(const self& lhs, const self& rhs) { return lhs.base == rhs.base; }

    reference operator*() const;
    pointer operator->() const;
    self& operator++();
    self operator++(int);
    self& operator--();
    self operator--(int);

    static int minusCount;
    friend constexpr difference_type operator-(SizedSentinelForwardIter const&a,
                                               SizedSentinelForwardIter const&b) {
      if (!std::is_constant_evaluated())
        minusCount++;
      return a.base - b.base;
    }
};

int SizedSentinelForwardIter::minusCount = 0;

struct ConvertibleForwardIter {
    typedef std::forward_iterator_tag       iterator_category;
    typedef int                             value_type;
    typedef std::ptrdiff_t                  difference_type;
    typedef int*                            pointer;
    typedef int&                            reference;
    typedef ConvertibleForwardIter          self;

    int *base_ = nullptr;

    constexpr ConvertibleForwardIter() = default;
    constexpr explicit ConvertibleForwardIter(int *ptr) : base_(ptr) { }

    friend bool operator==(const self&, const self&);

    reference operator*() const;
    pointer operator->() const;
    self& operator++();
    self operator++(int);
    self& operator--();
    self operator--(int);

    constexpr operator pointer() const { return base_; }

    // Explicitly deleted so this doesn't model sized_sentinel_for.
    friend constexpr difference_type operator-(int *, self const&) = delete;
    friend constexpr difference_type operator-(self const&, int*) = delete;
};
using ConvertibleForwardSubrange = std::ranges::subrange<ConvertibleForwardIter, int*,
                                                         std::ranges::subrange_kind::unsized>;
static_assert(std::is_convertible_v<ConvertibleForwardIter, int*>);

template<bool EnableConvertible>
struct ConditionallyConvertibleBase {
    typedef std::forward_iterator_tag            iterator_category;
    typedef int                                  value_type;
    typedef std::ptrdiff_t                       difference_type;
    typedef int*                                 pointer;
    typedef int&                                 reference;
    typedef std::make_unsigned_t<std::ptrdiff_t> udifference_type;
    typedef ConditionallyConvertibleBase         self;

    int *base_ = nullptr;

    constexpr ConditionallyConvertibleBase() = default;
    constexpr explicit ConditionallyConvertibleBase(int *ptr) : base_(ptr) {}

    constexpr int *base() const { return base_; }

    friend bool operator==(const self&, const self&) = default;

    reference operator*() const;
    pointer operator->() const;
    self& operator++();
    self operator++(int);
    self& operator--();
    self operator--(int);

    template<bool E = EnableConvertible>
      requires E
    constexpr operator pointer() const { return base_; }
};
using ConditionallyConvertibleIter = ConditionallyConvertibleBase<false>;
using SizedSentinelForwardSubrange = std::ranges::subrange<ConditionallyConvertibleIter,
                                                           ConditionallyConvertibleIter,
                                                           std::ranges::subrange_kind::sized>;
using ConvertibleSizedSentinelForwardIter = ConditionallyConvertibleBase<true>;
using ConvertibleSizedSentinelForwardSubrange = std::ranges::subrange<ConvertibleSizedSentinelForwardIter, int*,
                                                                      std::ranges::subrange_kind::sized>;

struct ForwardBorrowedRange {
  constexpr ForwardIter begin() const { return ForwardIter(globalBuff); }
  constexpr ForwardIter end() const { return ForwardIter(globalBuff + 8); }
};

template<>
inline constexpr bool std::ranges::enable_borrowed_range<ForwardBorrowedRange> = true;

struct ForwardRange {
  ForwardIter begin() const;
  ForwardIter end() const;
};

struct ConvertibleForwardBorrowedRange {
  constexpr ConvertibleForwardIter begin() const { return ConvertibleForwardIter(globalBuff); }
  constexpr int *end() const { return globalBuff + 8; }
};

template<>
inline constexpr bool std::ranges::enable_borrowed_range<ConvertibleForwardBorrowedRange> = true;

struct ForwardBorrowedRangeDifferentSentinel {
  struct sentinel {
    int *value;
    friend bool operator==(sentinel s, ForwardIter i) { return s.value == i.base(); }
  };

  constexpr ForwardIter begin() const { return ForwardIter(globalBuff); }
  constexpr sentinel end() const { return sentinel{globalBuff + 8}; }
};

template<>
inline constexpr bool std::ranges::enable_borrowed_range<ForwardBorrowedRangeDifferentSentinel> = true;

using DifferentSentinelSubrange = std::ranges::subrange<ForwardIter,
                                                        ForwardBorrowedRangeDifferentSentinel::sentinel,
                                                        std::ranges::subrange_kind::unsized>;

struct DifferentSentinelWithSizeMember {
  struct sentinel {
    int *value;
    friend bool operator==(sentinel s, ForwardIter i) { return s.value == i.base(); }
  };

  constexpr ForwardIter begin() const { return ForwardIter(globalBuff); }
  constexpr sentinel end() const { return sentinel{globalBuff + 8}; }
  constexpr size_t size() const { return 8; }
};

template<>
inline constexpr bool std::ranges::enable_borrowed_range<DifferentSentinelWithSizeMember> = true;

using DifferentSentinelWithSizeMemberSubrange = std::ranges::subrange<ForwardIter,
                                                                      DifferentSentinelWithSizeMember::sentinel,
                                                                      std::ranges::subrange_kind::unsized>;

#endif // SUBRANGE_TEST_TYPES_H
