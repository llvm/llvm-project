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

// This test makes sure that std::bit_cast fails when any of the following
// constraints are violated:
//
//      (1.1) sizeof(To) == sizeof(From) is true;
//      (1.2) is_trivially_copyable_v<To> is true;
//      (1.3) is_trivially_copyable_v<From> is true.
//
// Also check that it's ill-formed when the return type would be
// ill-formed, even though that is not explicitly mentioned in the
// specification (but it can be inferred from the synopsis).

#include <bit>
#include <concepts>

template<class To, class From>
concept bit_cast_is_valid = requires(From from) {
    { std::bit_cast<To>(from) } -> std::same_as<To>;
};

// Types are not the same size
namespace ns1 {
    struct To { char a; };
    struct From { char a; char b; };
    static_assert(!bit_cast_is_valid<To, From>);
    static_assert(!bit_cast_is_valid<From&, From>);
}

// To is not trivially copyable
namespace ns2 {
    struct To { char a; To(To const&); };
    struct From { char a; };
    static_assert(!bit_cast_is_valid<To, From>);
}

// From is not trivially copyable
namespace ns3 {
    struct To { char a; };
    struct From { char a; From(From const&); };
    static_assert(!bit_cast_is_valid<To, From>);
}

// The return type is ill-formed
namespace ns4 {
    struct From { char a; char b; };
    static_assert(!bit_cast_is_valid<char[2], From>);
    static_assert(!bit_cast_is_valid<int(), From>);
}
