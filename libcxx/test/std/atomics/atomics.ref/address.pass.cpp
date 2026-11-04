//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// TODO: Update test after https://llvm.org/PR121414 lands.
// ADDITIONAL_COMPILE_FLAGS(gcc-style-warnings): -Wno-deprecated-volatile
// ADDITIONAL_COMPILE_FLAGS: -Wno-deprecated-volatile

// constexpr address-return-type address() const noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <memory>
#include <type_traits>

#include "atomic_helpers.h"
#include "test_macros.h"

template <template <class TestArg> class TestFunctor, template <class> class AddQualifier>
struct TestEachAtomicTypeWithCV {
  template <class T>
  struct Qualified {
    void operator()() const { TestFunctor<AddQualifier<T>>()(); }
  };

  void operator()() const { TestEachAtomicType<Qualified>()(); }
};

template <template <class TestArg> class TestFunctor>
struct TestEachCVAtomicType {
  void operator()() const {
    TestEachAtomicTypeWithCV<TestFunctor, std::type_identity_t>()();
    TestEachAtomicTypeWithCV<TestFunctor, std::add_const_t>()();
    TestEachAtomicTypeWithCV<TestFunctor, std::add_volatile_t>()();
    TestEachAtomicTypeWithCV<TestFunctor, std::add_cv_t>()();
  }
};

template <typename T>
struct TestAddress {
  void operator()() const {
    alignas(std::atomic_ref<T>::required_alignment) T x(T(1));
    const std::atomic_ref<T> a(x);

    using AddressReturnT =
        std::conditional_t<std::is_const_v<T> && std::is_volatile_v<T>,
                           const volatile void,
                           std::conditional_t<std::is_volatile_v<T>,
                                              volatile void,
                                              std::conditional_t<std::is_const_v<T>, const void, void>>>*;

    std::same_as<AddressReturnT> decltype(auto) p = a.address();
    assert(std::addressof(x) == p);

    static_assert(noexcept(a.address()));
  }
};

int main(int, char**) {
  TestEachCVAtomicType<TestAddress>()();

  return 0;
}
