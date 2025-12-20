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
#include <type_traits>

inline bool called;
inline void call_func() noexcept { called = true; }

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
  int operator()(int i) const noexcept { return i; }
};

struct TriviallyDestructibleTooLarge {
  TriviallyDestructibleTooLarge() = default;
  TriviallyDestructibleTooLarge(MoveCounter) {}
  TriviallyDestructibleTooLarge(std::initializer_list<int>, MoveCounter) {}
  void operator()() const noexcept { called = true; }
  int operator()(int i) const noexcept { return i; }
  char a[5 * sizeof(void*)];
};

struct NonTrivial {
  NonTrivial() = default;
  NonTrivial(MoveCounter) {}
  NonTrivial(std::initializer_list<int>&, MoveCounter) {}
  NonTrivial(NonTrivial&&) noexcept(false) {}
  ~NonTrivial() {}

  void operator()() const noexcept { called = true; }
  int operator()(int i) const noexcept { return i; }
};

inline int get_val(int i) noexcept { return i; }

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
  void operator()() const& { *type = ConstLValue; }
  void operator()() const&& { *type = ConstRValue; }
};

struct CallTypeCheckerNoexcept {
  CallType* type;
  using enum CallType;
  void operator()() & noexcept { *type = LValue; }
  void operator()() && noexcept { *type = RValue; }
  void operator()() const& noexcept { *type = ConstLValue; }
  void operator()() const&& noexcept { *type = ConstRValue; }
};

#endif // MOVE_ONLY_FUNCTION_COMMON_H
