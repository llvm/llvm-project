//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_LIBCXX_UTILITIES_SMARTPTR_ADAPT_TYPES_H
#define TEST_LIBCXX_UTILITIES_SMARTPTR_ADAPT_TYPES_H

#include <type_traits>
#include <memory>

struct ResetArg {};

// Custom pointer types.

template <typename _Tp>
struct ConstructiblePtr {
  using pointer = _Tp*;
  std::unique_ptr<_Tp> ptr;

  ConstructiblePtr() = default;
  explicit ConstructiblePtr(_Tp* p) : ptr{p} {}

  auto operator==(_Tp val) { return *ptr == val; }

  auto* get() const { return ptr.get(); }

  void release() { ptr.release(); }
};

static_assert(std::is_same_v<std::__pointer_of_t< ConstructiblePtr<int>>, int* >);
static_assert(std::is_constructible_v< ConstructiblePtr<int>, int* >);

template <typename _Tp>
struct ResettablePtr {
  using element_type = _Tp;
  std::unique_ptr<_Tp> ptr;

  explicit ResettablePtr(_Tp* p) : ptr{p} {}

  auto operator*() const { return *ptr; }

  auto operator==(_Tp val) { return *ptr == val; }

  void reset() { ptr.reset(); }
  void reset(_Tp* p, ResetArg) { ptr.reset(p); }

  auto* get() const { return ptr.get(); }

  void release() { ptr.release(); }
};

static_assert(std::is_same_v<std::__pointer_of_t< ResettablePtr<int>>, int* >);
static_assert(std::is_constructible_v< ResettablePtr<int>, int* >);

template <typename _Tp>
struct NonConstructiblePtr : public ResettablePtr<_Tp> {
  NonConstructiblePtr() : NonConstructiblePtr::ResettablePtr(nullptr){};

  void reset(_Tp* p) { ResettablePtr<_Tp>::ptr.reset(p); }
};

static_assert(std::is_same_v<std::__pointer_of_t< NonConstructiblePtr<int>>, int* >);
static_assert(!std::is_constructible_v< NonConstructiblePtr<int>, int* >);

// Custom types.

struct SomeInt {
  int value;

  constexpr explicit SomeInt(int val = 0) : value{val} {}
};

#endif // TEST_LIBCXX_UTILITIES_SMARTPTR_ADAPT_TYPES_H
