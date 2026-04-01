//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// ADDITIONAL_COMPILE_FLAGS(gcc-style-warnings): -Wno-deprecated-volatile
// ADDITIONAL_COMPILE_FLAGS: -Wno-deprecated-volatile

// constexpr address-return-type address() const noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <memory>

#include "atomic_helpers.h"
#include "test_macros.h"

// Let COPYCV(FROM, TO) be an alias for type TO with the addition of FROM's
// top-level cv-qualifiers.
template <class _From>
struct copy_cv {
  template <class _To>
  using apply _LIBCPP_NODEBUG = _To;
};

template <class _From>
struct copy_cv<const _From> {
  template <class _To>
  using apply _LIBCPP_NODEBUG = const _To;
};

template <class _From>
struct copy_cv<volatile _From> {
  template <class _To>
  using apply _LIBCPP_NODEBUG = volatile _To;
};

template <class _From>
struct copy_cv<const volatile _From> {
  template <class _To>
  using apply _LIBCPP_NODEBUG = const volatile _To;
};

template <class _From, class _To>
using copy_cv_t _LIBCPP_NODEBUG = typename copy_cv<_From>::template apply<_To>;

template <class T>
using identity_t = T;

template <class T>
using add_const_t = const T;

template <class T>
using add_volatile_t = volatile T;

template <class T>
using add_const_volatile_t = const volatile T;

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

    using AddressReturnT = copy_cv_t<T, void>*;

    std::same_as<AddressReturnT> decltype(auto) p = a.address();
    assert(std::addressof(x) == p);

    static_assert(noexcept(a.address()));
  }
};

int main(int, char**) {
  TestEachCVAtomicType<TestAddress>()();

  return 0;
}
