//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

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

template <template <class TestArg> class TestFunctor>
struct TestEachCVAtomicType {
  void operator()() const {
    // integral types

    TestFunctor<char>()();
    TestFunctor<signed char>()();
    TestFunctor<unsigned char>()();
    TestFunctor<short>()();
    TestFunctor<unsigned short>()();
    TestFunctor<int>()();
    TestFunctor<unsigned int>()();
    TestFunctor<long>()();
    TestFunctor<unsigned long>()();
    TestFunctor<long long>()();
    TestFunctor<unsigned long long>()();
    TestFunctor<wchar_t>()();
#if TEST_STD_VER >= 20 && defined(__cpp_char8_t)
    TestFunctor<char8_t>()();
#endif
    TestFunctor<char16_t>()();
    TestFunctor<char32_t>()();
    TestFunctor<std::int8_t>()();
    TestFunctor<std::uint8_t>()();
    TestFunctor<std::int16_t>()();
    TestFunctor<std::uint16_t>()();
    TestFunctor<std::int32_t>()();
    TestFunctor<std::uint32_t>()();
    TestFunctor<std::int64_t>()();
    TestFunctor<std::uint64_t>()();

    TestFunctor<const char>()();
    TestFunctor<const signed char>()();
    TestFunctor<const unsigned char>()();
    TestFunctor<const short>()();
    TestFunctor<const unsigned short>()();
    TestFunctor<const int>()();
    TestFunctor<const unsigned int>()();
    TestFunctor<const long>()();
    TestFunctor<const unsigned long>()();
    TestFunctor<const long long>()();
    TestFunctor<const unsigned long long>()();
    TestFunctor<const wchar_t>()();
#if TEST_STD_VER >= 20 && defined(__cpp_char8_t)
    TestFunctor<const char8_t>()();
#endif
    TestFunctor<const char16_t>()();
    TestFunctor<const char32_t>()();
    TestFunctor<const std::int8_t>()();
    TestFunctor<const std::uint8_t>()();
    TestFunctor<const std::int16_t>()();
    TestFunctor<const std::uint16_t>()();
    TestFunctor<const std::int32_t>()();
    TestFunctor<const std::uint32_t>()();
    TestFunctor<const std::int64_t>()();
    TestFunctor<const std::uint64_t>()();

    TestFunctor<char>()();
    TestFunctor<signed char>()();
    TestFunctor<unsigned char>()();
    TestFunctor<short>()();
    TestFunctor<unsigned short>()();
    TestFunctor<int>()();
    TestFunctor<unsigned int>()();
    TestFunctor<long>()();
    TestFunctor<unsigned long>()();
    TestFunctor<long long>()();
    TestFunctor<unsigned long long>()();
    TestFunctor<wchar_t>()();
#if TEST_STD_VER >= 20 && defined(__cpp_char8_t)
    TestFunctor<char8_t>()();
#endif
    TestFunctor<char16_t>()();
    TestFunctor<char32_t>()();
    TestFunctor<std::int8_t>()();
    TestFunctor<std::uint8_t>()();
    TestFunctor<std::int16_t>()();
    TestFunctor<std::uint16_t>()();
    TestFunctor<std::int32_t>()();
    TestFunctor<std::uint32_t>()();
    TestFunctor<std::int64_t>()();
    TestFunctor<std::uint64_t>()();

    // floating-point types

    TestFunctor<float>()();
    TestFunctor<double>()();

    TestFunctor<const float>()();
    TestFunctor<const double>()();
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
