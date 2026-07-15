//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>
//
// template<class Duration>
// class hh_mm_ss {
// public:
//   constexpr explicit hh_mm_ss(Duration d);
// };
//
// LWG4274: The hh_mm_ss constructor supports unsigned durations.

#include <chrono>
#include <ratio>

// A signed arithmetic-like type used as a custom duration representation.
struct SignedRep {
  long long value;

  constexpr explicit SignedRep(long long v = 0) : value(v) {}
  constexpr operator long long() const { return value; }

  friend constexpr SignedRep operator-(SignedRep v) { return SignedRep{-v.value}; }
  friend constexpr bool operator<(SignedRep lhs, SignedRep rhs) { return lhs.value < rhs.value; }
};

int main(int, char**) {
  {
    // Tests construction from a duration with an unsigned representation.
    using Duration = std::chrono::duration<unsigned, std::milli>;

    // 1 hour + 1 minute + 1 second + 1 millisecond
    constexpr Duration d{3'661'001};
    constexpr std::chrono::hh_mm_ss<Duration> hms{d};

    static_assert(!hms.is_negative());
    static_assert(hms.hours() == std::chrono::hours{1});
    static_assert(hms.minutes() == std::chrono::minutes{1});
    static_assert(hms.seconds() == std::chrono::seconds{1});
    static_assert(hms.subseconds() == std::chrono::milliseconds{1});
    static_assert(hms.to_duration() == std::chrono::milliseconds{3'661'001});
  }

  {
    // Tests construction from a negative duration with a custom representation.
    using Duration = std::chrono::duration<SignedRep>;

    // 1 hour + 1 minute + 1 second
    constexpr Duration d{SignedRep{-3'661}};
    constexpr std::chrono::hh_mm_ss<Duration> hms{d};

    static_assert(hms.is_negative());
    static_assert(hms.hours() == std::chrono::hours{1});
    static_assert(hms.minutes() == std::chrono::minutes{1});
    static_assert(hms.seconds() == std::chrono::seconds{1});
    static_assert(hms.subseconds() == std::chrono::seconds{0});
    static_assert(hms.to_duration() == std::chrono::seconds{-3'661});
  }

  return 0;
}
