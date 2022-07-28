//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MOVE_ONLY_FUNCTION_COMMON_H
#define MOVE_ONLY_FUNCTION_COMMON_H

#include <initializer_list>

#include "test_macros.h"

inline bool called;
inline void call_func() noexcept { called = true; }

template <class...>
struct call_test_impl;

template <class ReturnT, class Functor, class... Args>
struct call_test_impl<ReturnT(Args...), Functor> {
  call_test_impl(Functor func) {
    func.template operator()<ReturnT(Args...)>();
    func.template operator()<ReturnT(Args...)&>();
    func.template operator()<ReturnT(Args...) &&>();
    func.template operator()<ReturnT(Args...) noexcept>();
    func.template operator()<ReturnT(Args...) & noexcept>();
    func.template operator()<ReturnT(Args...) && noexcept>();
    func.template operator()<ReturnT(Args...) const>();
    func.template operator()<ReturnT(Args...) const&>();
    func.template operator()<ReturnT(Args...) const&&>();
    func.template operator()<ReturnT(Args...) const noexcept>();
    func.template operator()<ReturnT(Args...) const & noexcept>();
    func.template operator()<ReturnT(Args...) const && noexcept>();
  }
};

template <class FuncT, class Functor>
void call_test(Functor f) {
  call_test_impl<FuncT, Functor>{f};
}

struct MoveCounter {
  int* counter_;
  MoveCounter(int* counter) : counter_(counter) {}
  MoveCounter(MoveCounter&& other) : counter_(other.counter_) { ++*counter_; }
};

struct TriviallyDestructible {
  TriviallyDestructible() = default;
  TriviallyDestructible(MoveCounter) {}
  TriviallyDestructible(std::initializer_list<int>, MoveCounter) {}
  void operator()() const noexcept { called = true; }
};

struct TriviallyDestructibleSqueezeFit {
  TriviallyDestructibleSqueezeFit() = default;
  TriviallyDestructibleSqueezeFit(MoveCounter) {}
  TriviallyDestructibleSqueezeFit(std::initializer_list<int>, MoveCounter) {}
  void operator()() const noexcept { called = true; }

private:
  [[maybe_unused]] void* ptr;
  [[maybe_unused]] char a[4 * sizeof(void*) - 1];
};

struct TriviallyDestructibleTooLarge {
  TriviallyDestructibleTooLarge() = default;
  TriviallyDestructibleTooLarge(MoveCounter) {}
  TriviallyDestructibleTooLarge(std::initializer_list<int>, MoveCounter) {}
  void operator()() const noexcept { called = true; }
  char a[5 * sizeof(void*)];
};

#ifdef TEST_COMPILER_CLANG
struct [[clang::trivial_abi]] TriviallyRelocatable {
  TriviallyRelocatable() = default;
  TriviallyRelocatable(MoveCounter) {}
  TriviallyRelocatable(std::initializer_list<int>, MoveCounter) {}
  TriviallyRelocatable(TriviallyRelocatable&&) {}
  ~TriviallyRelocatable() {}

  void operator()() const noexcept { called = true; }
};

struct [[clang::trivial_abi]] TriviallyRelocatableSqueezeFit {
  TriviallyRelocatableSqueezeFit() = default;
  TriviallyRelocatableSqueezeFit(MoveCounter) {}
  TriviallyRelocatableSqueezeFit(std::initializer_list<int>, MoveCounter) {}
  TriviallyRelocatableSqueezeFit(TriviallyRelocatableSqueezeFit&&) {}
  ~TriviallyRelocatableSqueezeFit() {}

  void operator()() const noexcept { called = true; }

private:
  [[maybe_unused]] void* ptr;
  [[maybe_unused]] char a[3 * sizeof(void*) - 1];
};

struct [[clang::trivial_abi]] TriviallyRelocatableTooLarge {
  TriviallyRelocatableTooLarge() = default;
  TriviallyRelocatableTooLarge(MoveCounter) {}
  TriviallyRelocatableTooLarge(std::initializer_list<int>, MoveCounter) {}
  ~TriviallyRelocatableTooLarge() {}

  void operator()() const noexcept { called = true; }

private:
  [[maybe_unused]] char a[4 * sizeof(void*)];
};
#endif

struct NonTrivial {
  NonTrivial() = default;
  NonTrivial(MoveCounter) {}
  NonTrivial(std::initializer_list<int>, MoveCounter) {}
  NonTrivial(NonTrivial&&) {}
  ~NonTrivial() {}

  void operator()() const noexcept { called = true; }
};

enum class CallType {
  None,
  LValue,
  RValue,
  ConstLValue,
  ConstRValue,
};

struct CallTypeChecker {
  CallType* type;
  using enum CallType;
  void operator()() & { *type = LValue; }
  void operator()() && { *type = RValue; }
  void operator()() const & { *type = ConstLValue; }
  void operator()() const && { *type = ConstRValue; }
};

struct CallTypeCheckerNoexcept {
  CallType* type;
  using enum CallType;
  void operator()() & noexcept { *type = LValue; }
  void operator()() && noexcept { *type = RValue; }
  void operator()() const & noexcept { *type = ConstLValue; }
  void operator()() const && noexcept { *type = ConstRValue; }
};

#endif // MOVE_ONLY_FUNCTION_COMMON_H
