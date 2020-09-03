//===-- include/flang/Common/interval.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_COMMON_INTERVAL_H_
#define FORTRAN_COMMON_INTERVAL_H_

// Defines a generalized template class Interval<A> to represent
// the half-open interval [x .. x+n).

#include "idioms.h"
#include <algorithm>
#include <cstddef>
#include <utility>

namespace Fortran::common {

template <typename A> class Interval {
public:
  using type = A;
  constexpr Interval() {}
  constexpr Interval(const A &s, std::size_t n = 1) : start_{s}, size_{n} {}
  constexpr Interval(A &&s, std::size_t n = 1)
      : start_{std::move(s)}, size_{n} {}
  constexpr Interval(const Interval &) = default;
  constexpr Interval(Interval &&) = default;
  constexpr Interval &operator=(const Interval &) = default;
  constexpr Interval &operator=(Interval &&) = default;

  constexpr bool operator<(const Interval &that) const {
    return start_ < that.start_ ||
        (start_ == that.start_ && size_ < that.size_);
  }
  constexpr bool operator<=(const Interval &that) const {
    return start_ < that.start_ ||
        (start_ == that.start_ && size_ <= that.size_);
  }
  constexpr bool operator==(const Interval &that) const {
    return start_ == that.start_ && size_ == that.size_;
  }
  constexpr bool operator!=(const Interval &that) const {
    return !(*this == that);
  }
  constexpr bool operator>=(const Interval &that) const {
    return !(*this < that);
  }
  constexpr bool operator>(const Interval &that) const {
    return !(*this <= that);
  }

  constexpr const A &start() const { return start_; }
  constexpr std::size_t size() const { return size_; }
  constexpr bool empty() const { return size_ == 0; }

  constexpr bool Contains(const A &x) const {
    return start_ <= x && x < start_ + size_;
  }
  constexpr bool Contains(const Interval &that) const {
    return Contains(that.start_) && Contains(that.start_ + (that.size_ - 1));
  }
  constexpr bool IsDisjointWith(const Interval &that) const {
    return that.NextAfter() <= start_ || NextAfter() <= that.start_;
  }
  constexpr bool ImmediatelyPrecedes(const Interval &that) const {
    return NextAfter() == that.start_;
  }
  void Annex(const Interval &that) {
    size_ = (that.start_ + that.size_) - start_;
  }
  bool AnnexIfPredecessor(const Interval &that) {
    if (ImmediatelyPrecedes(that)) {
      size_ += that.size_;
      return true;
    }
    return false;
  }
  void ExtendToCover(const Interval &that) {
    if (size_ == 0) {
      *this = that;
    } else if (that.size_ != 0) {
      const auto end{std::max(NextAfter(), that.NextAfter())};
      start_ = std::min(start_, that.start_);
      size_ = end - start_;
    }
  }

  std::size_t MemberOffset(const A &x) const {
    CHECK(Contains(x));
    return x - start_;
  }
  A OffsetMember(std::size_t n) const {
    CHECK(n < size_);
    return start_ + n;
  }

  constexpr A Last() const { return start_ + (size_ - 1); }
  constexpr A NextAfter() const { return start_ + size_; }
  constexpr Interval Prefix(std::size_t n) const {
    return {start_, std::min(size_, n)};
  }
  Interval Suffix(std::size_t n) const {
    CHECK(n <= size_);
    return {start_ + n, size_ - n};
  }

  constexpr Interval Intersection(const Interval &that) const {
    if (that.NextAfter() <= start_) {
      return {};
    } else if (that.start_ <= start_) {
      auto skip{start_ - that.start_};
      return {start_, std::min(size_, that.size_ - skip)};
    } else if (NextAfter() <= that.start_) {
      return {};
    } else {
      auto skip{that.start_ - start_};
      return {that.start_, std::min(that.size_, size_ - skip)};
    }
  }

private:
  A start_;
  std::size_t size_{0};
};
} // namespace Fortran::common
#endif // FORTRAN_COMMON_INTERVAL_H_
