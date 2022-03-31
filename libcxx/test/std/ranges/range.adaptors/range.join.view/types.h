//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_JOIN_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_JOIN_TYPES_H

#include <concepts>

#include "test_macros.h"
#include "test_iterators.h"

int globalBuffer[4][4] = {{1111, 2222, 3333, 4444}, {555, 666, 777, 888}, {99, 1010, 1111, 1212}, {13, 14, 15, 16}};

struct ChildView : std::ranges::view_base {
  int *ptr_;

  constexpr ChildView(int *ptr = globalBuffer[0]) : ptr_(ptr) {}
  ChildView(const ChildView&) = delete;
  ChildView(ChildView&&) = default;
  ChildView& operator=(const ChildView&) = delete;
  ChildView& operator=(ChildView&&) = default;

  constexpr cpp20_input_iterator<int *> begin() { return cpp20_input_iterator<int *>(ptr_); }
  constexpr cpp20_input_iterator<const int *> begin() const { return cpp20_input_iterator<const int *>(ptr_); }
  constexpr int *end() { return ptr_ + 4; }
  constexpr const int *end() const { return ptr_ + 4; }
};

constexpr bool operator==(const cpp20_input_iterator<int*> &lhs, int* rhs) { return base(lhs) == rhs; }
constexpr bool operator==(int* lhs, const cpp20_input_iterator<int*> &rhs) { return base(rhs) == lhs; }

ChildView globalChildren[4] = {ChildView(globalBuffer[0]), ChildView(globalBuffer[1]), ChildView(globalBuffer[2]), ChildView(globalBuffer[3])};

template<class T>
struct ParentView : std::ranges::view_base {
  T *ptr_;
  unsigned size_;

  constexpr ParentView(T *ptr, unsigned size = 4)
    : ptr_(ptr), size_(size) {}
  constexpr ParentView(ChildView *ptr = globalChildren, unsigned size = 4)
    requires std::same_as<ChildView, T>
    : ptr_(ptr), size_(size) {}
  ParentView(const ParentView&) = delete;
  ParentView(ParentView&&) = default;
  ParentView& operator=(const ParentView&) = delete;
  ParentView& operator=(ParentView&&) = default;

  constexpr cpp20_input_iterator<T *> begin() { return cpp20_input_iterator<T *>(ptr_); }
  constexpr cpp20_input_iterator<const T *> begin() const { return cpp20_input_iterator<const T *>(ptr_); }
  constexpr T *end() { return ptr_ + size_; }
  constexpr const T *end() const { return ptr_ + size_; }
};
// TODO: remove these bogus operators
template<class T>
constexpr bool operator==(const cpp20_input_iterator<T*> &lhs, T *rhs) { return base(lhs) == rhs; }
template<class T>
constexpr bool operator==(T *lhs, const cpp20_input_iterator<T*> &rhs) { return base(rhs) == lhs; }

struct CopyableChild : std::ranges::view_base {
  int *ptr_;
  unsigned size_;
  constexpr CopyableChild(int *ptr = globalBuffer[0], unsigned size = 4)
    : ptr_(ptr), size_(size) {}

  constexpr cpp17_input_iterator<int *> begin() { return cpp17_input_iterator<int *>(ptr_); }
  constexpr cpp17_input_iterator<const int *> begin() const { return cpp17_input_iterator<const int *>(ptr_); }
  constexpr int *end() { return ptr_ + size_; }
  constexpr const int *end() const { return ptr_ + size_; }
};
// TODO: remove these bogus operators
constexpr bool operator==(const cpp17_input_iterator<const int*> &lhs, const int* rhs) { return base(lhs) == rhs; }
constexpr bool operator==(const int* lhs, const cpp17_input_iterator<const int*> &rhs) { return base(rhs) == lhs; }

struct CopyableParent : std::ranges::view_base {
  CopyableChild *ptr_;
  constexpr CopyableParent(CopyableChild *ptr) : ptr_(ptr) {}

  constexpr cpp17_input_iterator<CopyableChild *> begin() { return cpp17_input_iterator<CopyableChild *>(ptr_); }
  constexpr cpp17_input_iterator<const CopyableChild *> begin() const { return cpp17_input_iterator<const CopyableChild *>(ptr_); }
  constexpr CopyableChild *end() { return ptr_ + 4; }
  constexpr const CopyableChild *end() const { return ptr_ + 4; }
};
// TODO: remove these bogus operators
constexpr bool operator==(const cpp17_input_iterator<const CopyableChild*> &lhs, const CopyableChild *rhs) { return base(lhs) == rhs; }
constexpr bool operator==(const CopyableChild *lhs, const cpp17_input_iterator<const CopyableChild*> &rhs) { return base(rhs) == lhs; }

struct Box { int x; };

template<class T>
struct InputValueIter {
  typedef std::input_iterator_tag iterator_category;
  typedef T value_type;
  typedef int difference_type;
  typedef T reference;

  T *ptr_;
  constexpr InputValueIter(T *ptr) : ptr_(ptr) {}

  constexpr T operator*() const { return std::move(*ptr_); }
  constexpr void operator++(int) { ++ptr_; }
  constexpr InputValueIter& operator++() { ++ptr_; return *this; }

  constexpr T *operator->() { return ptr_; }
};

template<class T>
constexpr bool operator==(const InputValueIter<T> &lhs, const T* rhs) { return lhs.ptr_ == rhs; }
template<class T>
constexpr bool operator==(const T* lhs, const InputValueIter<T> &rhs) { return rhs.ptr_ == lhs; }

template<class T>
struct ValueView : std::ranges::view_base {
  InputValueIter<T> ptr_;

  constexpr ValueView(T *ptr) : ptr_(ptr) {}

  constexpr ValueView(ValueView &&other)
    : ptr_(other.ptr_) { other.ptr_.ptr_ = nullptr; }

  constexpr ValueView& operator=(ValueView &&other) {
    ptr_ = other.ptr_;
    other.ptr_ = InputValueIter<T>(nullptr);
    return *this;
  }

  ValueView(const ValueView&) = delete;
  ValueView& operator=(const ValueView&) = delete;

  constexpr InputValueIter<T> begin() { return ptr_; }
  constexpr InputValueIter<T> begin() const { return ptr_; }
  constexpr T *end() { return ptr_.ptr_ + 4; }
  constexpr const T *end() const { return ptr_.ptr_ + 4; }
};

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_JOIN_TYPES_H
