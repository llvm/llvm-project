//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <bit>
//
// template<class To, class From>
//   constexpr To bit_cast(const From& from) noexcept; // C++20

#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "test_macros.h"

// std::bit_cast does not preserve padding bits, so if T has padding bits,
// the results might not memcmp cleanly.
template<bool HasUniqueObjectRepresentations = true, typename T>
void test_roundtrip_through_buffer(T from) {
    struct Buffer { char buffer[sizeof(T)]; };
    Buffer middle = std::bit_cast<Buffer>(from);
    T to = std::bit_cast<T>(middle);
    Buffer middle2 = std::bit_cast<Buffer>(to);

    assert((from == to) == (from == from)); // because NaN

    if constexpr (HasUniqueObjectRepresentations) {
        assert(std::memcmp(&from, &middle, sizeof(T)) == 0);
        assert(std::memcmp(&to, &middle, sizeof(T)) == 0);
        assert(std::memcmp(&middle, &middle2, sizeof(T)) == 0);
    }
}

template<bool HasUniqueObjectRepresentations = true, typename T>
void test_roundtrip_through_nested_T(T from) {
    struct Nested { T x; };
    static_assert(sizeof(Nested) == sizeof(T));

    Nested middle = std::bit_cast<Nested>(from);
    T to = std::bit_cast<T>(middle);
    Nested middle2 = std::bit_cast<Nested>(to);

    assert((from == to) == (from == from)); // because NaN

    if constexpr (HasUniqueObjectRepresentations) {
        assert(std::memcmp(&from, &middle, sizeof(T)) == 0);
        assert(std::memcmp(&to, &middle, sizeof(T)) == 0);
        assert(std::memcmp(&middle, &middle2, sizeof(T)) == 0);
    }
}

template <typename Intermediate, bool HasUniqueObjectRepresentations = true, typename T>
void test_roundtrip_through(T from) {
    static_assert(sizeof(Intermediate) == sizeof(T));

    Intermediate middle = std::bit_cast<Intermediate>(from);
    T to = std::bit_cast<T>(middle);
    Intermediate middle2 = std::bit_cast<Intermediate>(to);

    assert((from == to) == (from == from)); // because NaN

    if constexpr (HasUniqueObjectRepresentations) {
        assert(std::memcmp(&from, &middle, sizeof(T)) == 0);
        assert(std::memcmp(&to, &middle, sizeof(T)) == 0);
        assert(std::memcmp(&middle, &middle2, sizeof(T)) == 0);
    }
}

template <typename T>
constexpr std::array<T, 10> generate_signed_integral_values() {
    return {std::numeric_limits<T>::min(),
            std::numeric_limits<T>::min() + 1,
            static_cast<T>(-2), static_cast<T>(-1),
            static_cast<T>(0), static_cast<T>(1),
            static_cast<T>(2), static_cast<T>(3),
            std::numeric_limits<T>::max() - 1,
            std::numeric_limits<T>::max()};
}

template <typename T>
constexpr std::array<T, 6> generate_unsigned_integral_values() {
    return {static_cast<T>(0), static_cast<T>(1),
            static_cast<T>(2), static_cast<T>(3),
            std::numeric_limits<T>::max() - 1,
            std::numeric_limits<T>::max()};
}

bool tests() {
    for (bool b : {false, true}) {
        test_roundtrip_through_nested_T(b);
        test_roundtrip_through_buffer(b);
        test_roundtrip_through<char>(b);
    }

    for (char c : {'\0', 'a', 'b', 'c', 'd'}) {
        test_roundtrip_through_nested_T(c);
        test_roundtrip_through_buffer(c);
    }

    // Fundamental signed integer types
    for (signed char i : generate_signed_integral_values<signed char>()) {
        test_roundtrip_through_nested_T(i);
        test_roundtrip_through_buffer(i);
    }

    for (short i : generate_signed_integral_values<short>()) {
        test_roundtrip_through_nested_T(i);
        test_roundtrip_through_buffer(i);
    }

    for (int i : generate_signed_integral_values<int>()) {
        test_roundtrip_through_nested_T(i);
        test_roundtrip_through_buffer(i);
        test_roundtrip_through<float>(i);
    }

    for (long i : generate_signed_integral_values<long>()) {
        test_roundtrip_through_nested_T(i);
        test_roundtrip_through_buffer(i);
    }

    for (long long i : generate_signed_integral_values<long long>()) {
        test_roundtrip_through_nested_T(i);
        test_roundtrip_through_buffer(i);
        test_roundtrip_through<double>(i);
    }

    // Fundamental unsigned integer types
    for (unsigned char i : generate_unsigned_integral_values<unsigned char>()) {
        test_roundtrip_through_nested_T(i);
        test_roundtrip_through_buffer(i);
    }

    for (unsigned short i : generate_unsigned_integral_values<unsigned short>()) {
        test_roundtrip_through_nested_T(i);
        test_roundtrip_through_buffer(i);
    }

    for (unsigned int i : generate_unsigned_integral_values<unsigned int>()) {
        test_roundtrip_through_nested_T(i);
        test_roundtrip_through_buffer(i);
        test_roundtrip_through<float>(i);
    }

    for (unsigned long i : generate_unsigned_integral_values<unsigned long>()) {
        test_roundtrip_through_nested_T(i);
        test_roundtrip_through_buffer(i);
    }

    for (unsigned long long i : generate_unsigned_integral_values<unsigned long long>()) {
        test_roundtrip_through_nested_T(i);
        test_roundtrip_through_buffer(i);
        test_roundtrip_through<double>(i);
    }

    // Fixed width signed integer types
    for (std::int32_t i : generate_signed_integral_values<std::int32_t>()) {
        test_roundtrip_through_nested_T(i);
        test_roundtrip_through_buffer(i);
        test_roundtrip_through<int>(i);
        test_roundtrip_through<std::uint32_t>(i);
        test_roundtrip_through<float>(i);
    }

    for (std::int64_t i : generate_signed_integral_values<std::int64_t>()) {
        test_roundtrip_through_nested_T(i);
        test_roundtrip_through_buffer(i);
        test_roundtrip_through<long long>(i);
        test_roundtrip_through<std::uint64_t>(i);
        test_roundtrip_through<double>(i);
    }

    // Fixed width unsigned integer types
    for (std::uint32_t i : generate_unsigned_integral_values<std::uint32_t>()) {
        test_roundtrip_through_nested_T(i);
        test_roundtrip_through_buffer(i);
        test_roundtrip_through<int>(i);
        test_roundtrip_through<std::int32_t>(i);
        test_roundtrip_through<float>(i);
    }

    for (std::uint64_t i : generate_unsigned_integral_values<std::uint64_t>()) {
        test_roundtrip_through_nested_T(i);
        test_roundtrip_through_buffer(i);
        test_roundtrip_through<long long>(i);
        test_roundtrip_through<std::int64_t>(i);
        test_roundtrip_through<double>(i);
    }

    // Floating point types
    for (float i : {0.0f, 1.0f, -1.0f, 10.0f, -10.0f, 1e10f, 1e-10f, 1e20f, 1e-20f, 2.71828f, 3.14159f,
                    std::nanf(""),
                    __builtin_nanf("0x55550001"), // NaN with a payload
                    std::numeric_limits<float>::signaling_NaN(),
                    std::numeric_limits<float>::quiet_NaN()}) {
        test_roundtrip_through_nested_T(i);
        test_roundtrip_through_buffer(i);
        test_roundtrip_through<int>(i);
    }

    for (double i : {0.0, 1.0, -1.0, 10.0, -10.0, 1e10, 1e-10, 1e100, 1e-100,
                     2.718281828459045,
                     3.141592653589793238462643383279502884197169399375105820974944,
                     std::nan(""),
                     std::numeric_limits<double>::signaling_NaN(),
                     std::numeric_limits<double>::quiet_NaN()}) {
        test_roundtrip_through_nested_T(i);
        test_roundtrip_through_buffer(i);
        test_roundtrip_through<long long>(i);
    }

    for (long double i : {0.0l, 1.0l, -1.0l, 10.0l, -10.0l, 1e10l, 1e-10l, 1e100l, 1e-100l,
                          2.718281828459045l,
                          3.141592653589793238462643383279502884197169399375105820974944l,
                          std::nanl(""),
                          std::numeric_limits<long double>::signaling_NaN(),
                          std::numeric_limits<long double>::quiet_NaN()}) {
        // Note that x86's `long double` has 80 value bits and 48 padding bits.
        test_roundtrip_through_nested_T<false>(i);
        test_roundtrip_through_buffer<false>(i);

#if __SIZEOF_LONG_DOUBLE__ == __SIZEOF_DOUBLE__
        test_roundtrip_through<double, false>(i);
#endif
#if defined(__SIZEOF_INT128__) && __SIZEOF_LONG_DOUBLE__ == __SIZEOF_INT128__ &&                                       \
    !TEST_HAS_FEATURE(memory_sanitizer) // Some bits are just padding.
        test_roundtrip_through<__int128_t, false>(i);
        test_roundtrip_through<__uint128_t, false>(i);
#endif
    }

    return true;
}

// TODO: There doesn't seem to be a way to perform non-trivial correctness
//       tests inside constexpr.
constexpr bool basic_constexpr_test() {
    struct Nested { char buffer[sizeof(int)]; };
    int from = 3;
    Nested middle = std::bit_cast<Nested>(from);
    int to = std::bit_cast<int>(middle);
    assert(from == to);
    return true;
}

int main(int, char**) {
    tests();
    static_assert(basic_constexpr_test());
    return 0;
}
