//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_CONCAT_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_CONCAT_TYPES_H

#include <ranges>
#include <utility>
#include "test_iterators.h"

struct TrackInitialization {
  constexpr explicit TrackInitialization(bool* moved, bool* copied) : moved_(moved), copied_(copied) {}
  constexpr TrackInitialization(TrackInitialization const& other) : moved_(other.moved_), copied_(other.copied_) {
    *copied_ = true;
  }
  constexpr TrackInitialization(TrackInitialization&& other) : moved_(other.moved_), copied_(other.copied_) {
    *moved_ = true;
  }
  TrackInitialization& operator=(TrackInitialization const&) = default;
  TrackInitialization& operator=(TrackInitialization&&)      = default;
  bool* moved_;
  bool* copied_;
};

template <class Iter, class Sent>
struct minimal_view : std::ranges::view_base {
  constexpr explicit minimal_view(Iter it, Sent sent) : it_(base(std::move(it))), sent_(base(std::move(sent))) {}

  minimal_view(minimal_view&&)            = default;
  minimal_view& operator=(minimal_view&&) = default;

  constexpr Iter begin() const { return Iter(it_); }
  constexpr Sent end() const { return Sent(sent_); }

private:
  decltype(base(std::declval<Iter>())) it_;
  decltype(base(std::declval<Sent>())) sent_;
};

template <class T>
struct BufferView : std::ranges::view_base {
  T* buffer_;
  std::size_t size_;

  template <std::size_t N>
  constexpr BufferView(T (&b)[N]) : buffer_(b), size_(N) {}
};

using IntBufferView = BufferView<int>;

template <bool Simple>
struct Common : IntBufferView {
  using IntBufferView::IntBufferView;

  constexpr int* begin()
    requires(!Simple)
  {
    return buffer_;
  }
  constexpr const int* begin() const { return buffer_; }
  constexpr int* end()
    requires(!Simple)
  {
    return buffer_ + size_;
  }
  constexpr const int* end() const { return buffer_ + size_; }
};

using SimpleCommon    = Common<true>;
using NonSimpleCommon = Common<false>;

using SimpleCommonRandomAccessSized    = SimpleCommon;
using NonSimpleCommonRandomAccessSized = NonSimpleCommon;

template <bool Simple>
struct NonCommon : IntBufferView {
  using IntBufferView::IntBufferView;
  constexpr int* begin()
    requires(!Simple)
  {
    return buffer_;
  }
  constexpr const int* begin() const { return buffer_; }
  constexpr sentinel_wrapper<int*> end()
    requires(!Simple)
  {
    return sentinel_wrapper<int*>(buffer_ + size_);
  }
  constexpr sentinel_wrapper<const int*> end() const { return sentinel_wrapper<const int*>(buffer_ + size_); }
};

using SimpleNonCommon    = NonCommon<true>;
using NonSimpleNonCommon = NonCommon<false>;

template <bool Simple>
struct NonCommonSized : IntBufferView {
  using IntBufferView::IntBufferView;
  constexpr int* begin()
    requires(!Simple)
  {
    return buffer_;
  }
  constexpr const int* begin() const { return buffer_; }
  constexpr sentinel_wrapper<int*> end()
    requires(!Simple)
  {
    return sentinel_wrapper<int*>(buffer_ + size_);
  }
  constexpr sentinel_wrapper<const int*> end() const { return sentinel_wrapper<const int*>(buffer_ + size_); }
  constexpr std::size_t size() const { return size_; }
};

using SimpleNonCommonSized                = NonCommonSized<true>;
using SimpleNonCommonRandomAccessSized    = SimpleNonCommonSized;
using NonSimpleNonCommonSized             = NonCommonSized<false>;
using NonSimpleNonCommonRandomAccessSized = NonSimpleNonCommonSized;

template <bool Simple>
struct NonCommonNonRandom : IntBufferView {
  using IntBufferView::IntBufferView;

  using const_iterator = forward_iterator<const int*>;
  using iterator       = forward_iterator<int*>;

  constexpr iterator begin()
    requires(!Simple)
  {
    return iterator(buffer_);
  }
  constexpr const_iterator begin() const { return const_iterator(buffer_); }
  constexpr sentinel_wrapper<iterator> end()
    requires(!Simple)
  {
    return sentinel_wrapper<iterator>(iterator(buffer_ + size_));
  }
  constexpr sentinel_wrapper<const_iterator> end() const {
    return sentinel_wrapper<const_iterator>(const_iterator(buffer_ + size_));
  }
};

using SimpleNonCommonNonRandom    = NonCommonNonRandom<true>;
using NonSimpleNonCommonNonRandom = NonCommonNonRandom<false>;

template <class Iter, class Sent = Iter, class NonConstIter = Iter, class NonConstSent = Sent>
struct BasicView : IntBufferView {
  using IntBufferView::IntBufferView;

  constexpr NonConstIter begin()
    requires(!std::is_same_v<Iter, NonConstIter>)
  {
    return NonConstIter(buffer_);
  }
  constexpr Iter begin() const { return Iter(buffer_); }

  constexpr NonConstSent end()
    requires(!std::is_same_v<Sent, NonConstSent>)
  {
    if constexpr (std::is_same_v<NonConstIter, NonConstSent>) {
      return NonConstIter(buffer_ + size_);
    } else {
      return NonConstSent(NonConstIter(buffer_ + size_));
    }
  }

  constexpr Sent end() const {
    if constexpr (std::is_same_v<Iter, Sent>) {
      return Iter(buffer_ + size_);
    } else {
      return Sent(Iter(buffer_ + size_));
    }
  }
};

using NonSizedRandomAccessView =
    BasicView<random_access_iterator<int*>, sentinel_wrapper<random_access_iterator<int*>>>;

using InputCommonView = BasicView<common_input_iterator<int*>>;

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_CONCAT_FILTER_TYPES_H
