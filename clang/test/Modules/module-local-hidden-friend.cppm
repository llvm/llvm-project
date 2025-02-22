// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -emit-module-interface -o %t/b.pcm \
// RUN:     -fmodule-file=a=%t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cc -fmodule-file=a=%t/a.pcm -fmodule-file=b=%t/b.pcm \
// RUN:     -fsyntax-only -verify
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-reduced-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -emit-reduced-module-interface -o %t/b.pcm \
// RUN:     -fmodule-file=a=%t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cc -fmodule-file=a=%t/a.pcm -fmodule-file=b=%t/b.pcm \
// RUN:     -fsyntax-only -verify

//--- a.cppm
export module a;

namespace n {
}

//--- ordering.mock.h
namespace std {
  class strong_ordering {
  public:
    int n;
    static const strong_ordering less, equal, greater;
    constexpr bool operator==(int n) const noexcept { return this->n == n;}
    constexpr bool operator!=(int n) const noexcept { return this->n != n;}
  };
  constexpr strong_ordering strong_ordering::less = {-1};
  constexpr strong_ordering strong_ordering::equal = {0};
  constexpr strong_ordering strong_ordering::greater = {1};

  class partial_ordering {
  public:
    long n;
    static const partial_ordering less, equal, greater, equivalent, unordered;
    constexpr bool operator==(long n) const noexcept { return this->n == n;}
    constexpr bool operator!=(long n) const noexcept { return this->n != n;}
  };
  constexpr partial_ordering partial_ordering::less = {-1};
  constexpr partial_ordering partial_ordering::equal = {0};
  constexpr partial_ordering partial_ordering::greater = {1};
  constexpr partial_ordering partial_ordering::equivalent = {0};
  constexpr partial_ordering partial_ordering::unordered = {-127};
} // namespace std

//--- b.cppm
module;
#include "ordering.mock.h"
export module b;

import a;

namespace n {

struct monostate {
	friend constexpr bool operator==(monostate, monostate) = default;
};

export struct wrapper {
	friend constexpr bool operator==(wrapper const &LHS, wrapper const &RHS) {
        return LHS.m_value == RHS.m_value;
    }

	monostate m_value;
};

struct monostate2 {
	auto operator<=>(monostate2 const &) const & = default;
};

export struct wrapper2 {
	friend bool operator==(wrapper2 const &LHS, wrapper2 const &RHS) = default;

	monostate2 m_value;
};

} // namespace n

//--- use.cc
// expected-no-diagnostics
import b;

static_assert(n::wrapper() == n::wrapper());
static_assert(n::wrapper2() == n::wrapper2());
