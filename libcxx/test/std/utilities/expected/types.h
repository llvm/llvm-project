//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_UTILITIES_EXPECTED_TYPES_H
#define TEST_STD_UTILITIES_EXPECTED_TYPES_H

#include <utility>
#include "test_macros.h"

template <bool copyMoveNoexcept, bool convertNoexcept = true>
struct TracedBase {
  struct state {
    bool copyCtorCalled   = false;
    bool copyAssignCalled = false;
    bool moveCtorCalled   = false;
    bool moveAssignCalled = false;
    bool dtorCalled       = false;
  };

  state* state_      = nullptr;
  bool copiedFromInt = false;
  bool movedFromInt  = false;
  bool copiedFromTmp = false;
  bool movedFromTmp  = false;
  int data_;

  constexpr TracedBase(const int& ii) noexcept(convertNoexcept) : data_(ii) { copiedFromInt = true; }
  constexpr TracedBase(int&& ii) noexcept(convertNoexcept) : data_(ii) { movedFromInt = true; }
  constexpr TracedBase(state& s, int ii) noexcept : state_(&s), data_(ii) {}
  constexpr TracedBase(const TracedBase& other) noexcept(copyMoveNoexcept) : state_(other.state_), data_(other.data_) {
    if (state_) {
      state_->copyCtorCalled = true;
    } else {
      copiedFromTmp = true;
    }
  }
  constexpr TracedBase(TracedBase&& other) noexcept(copyMoveNoexcept) : state_(other.state_), data_(other.data_) {
    if (state_) {
      state_->moveCtorCalled = true;
    } else {
      movedFromTmp = true;
    }
  }
  constexpr TracedBase& operator=(const TracedBase& other) noexcept(copyMoveNoexcept) {
    data_                    = other.data_;
    state_->copyAssignCalled = true;
    return *this;
  }
  constexpr TracedBase& operator=(TracedBase&& other) noexcept(copyMoveNoexcept) {
    data_                    = other.data_;
    state_->moveAssignCalled = true;
    return *this;
  }
  constexpr ~TracedBase() {
    if (state_) {
      state_->dtorCalled = true;
    }
  }
};

using Traced         = TracedBase<false>;
using TracedNoexcept = TracedBase<true>;

using MoveThrowConvNoexcept = TracedBase<false, true>;
using MoveNoexceptConvThrow = TracedBase<true, false>;
using BothMayThrow          = TracedBase<false, false>;
using BothNoexcept          = TracedBase<true, true>;

struct ADLSwap {
  int i;
  bool adlSwapCalled = false;
  constexpr ADLSwap(int ii) : i(ii) {}
  constexpr friend void swap(ADLSwap& x, ADLSwap& y) {
    std::swap(x.i, y.i);
    x.adlSwapCalled = true;
    y.adlSwapCalled = true;
  }
};

template <bool Noexcept>
struct TrackedMove {
  int i;
  int numberOfMoves = 0;
  bool swapCalled   = false;

  constexpr TrackedMove(int ii) : i(ii) {}
  constexpr TrackedMove(TrackedMove&& other) noexcept(Noexcept)
      : i(other.i), numberOfMoves(other.numberOfMoves), swapCalled(other.swapCalled) {
    ++numberOfMoves;
  }

  constexpr friend void swap(TrackedMove& x, TrackedMove& y) {
    std::swap(x.i, y.i);
    std::swap(x.numberOfMoves, y.numberOfMoves);
    x.swapCalled = true;
    y.swapCalled = true;
  }
};

#ifndef TEST_HAS_NO_EXCEPTIONS
struct Except {};

struct ThrowOnCopyConstruct {
  ThrowOnCopyConstruct() = default;
  ThrowOnCopyConstruct(const ThrowOnCopyConstruct&) { throw Except{}; }
  ThrowOnCopyConstruct& operator=(const ThrowOnCopyConstruct&) = default;
};

struct ThrowOnMoveConstruct {
  ThrowOnMoveConstruct() = default;
  ThrowOnMoveConstruct(ThrowOnMoveConstruct&&) { throw Except{}; }
  ThrowOnMoveConstruct& operator=(ThrowOnMoveConstruct&&) = default;
};

struct ThrowOnConvert {
  ThrowOnConvert() = default;
  ThrowOnConvert(const int&) { throw Except{}; }
  ThrowOnConvert(int&&) { throw Except{}; }
  ThrowOnConvert(const ThrowOnConvert&) noexcept(false) {}
  ThrowOnConvert& operator=(const ThrowOnConvert&) = default;
  ThrowOnConvert(ThrowOnConvert&&) noexcept(false) {}
  ThrowOnConvert& operator=(ThrowOnConvert&&) = default;
};

struct ThrowOnMove {
  bool* destroyed = nullptr;
  ThrowOnMove()   = default;
  ThrowOnMove(bool& d) : destroyed(&d) {}
  ThrowOnMove(ThrowOnMove&&) { throw Except{}; };
  ThrowOnMove& operator=(ThrowOnMove&&) = default;
  ~ThrowOnMove() {
    if (destroyed) {
      *destroyed = true;
    }
  }
};

#endif // TEST_HAS_NO_EXCEPTIONS

struct MoveOnlyErrorType {
  constexpr MoveOnlyErrorType(int) {}
  MoveOnlyErrorType(MoveOnlyErrorType&&) {}
  MoveOnlyErrorType(const MoveOnlyErrorType&&) {}
  MoveOnlyErrorType(const MoveOnlyErrorType&)            = delete;
  MoveOnlyErrorType& operator=(const MoveOnlyErrorType&) = delete;
};

#endif // TEST_STD_UTILITIES_EXPECTED_TYPES_H
