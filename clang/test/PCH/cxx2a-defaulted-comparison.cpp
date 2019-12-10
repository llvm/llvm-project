// RxN: %clang_cc1 -std=c++2a -verify -Wno-defaulted-function-deleted -include %s %s
//
// RUN: %clang_cc1 -std=c++2a -emit-pch %s -o %t.pch
// RUN: %clang_cc1 -std=c++2a -include-pch %t.pch %s -verify

// expected-no-diagnostics

#ifndef INCLUDED
#define INCLUDED

namespace std {
  struct strong_ordering {
    int n;
    constexpr operator int() const { return n; }
    static const strong_ordering equal, greater, less;
  };
  constexpr strong_ordering strong_ordering::equal = {0};
  constexpr strong_ordering strong_ordering::greater = {1};
  constexpr strong_ordering strong_ordering::less = {-1};
}

// Ensure that we can round-trip DefaultedFunctionInfo through an AST file.
namespace LookupContext {
  struct A {};

  namespace N {
    template <typename T> auto f() {
      bool operator==(const T &, const T &);
      bool operator<(const T &, const T &);
      struct B {
        T a;
        std::strong_ordering operator<=>(const B &) const = default;
      };
      return B();
    }
  }
}

#else

namespace LookupContext {
  namespace M {
    bool operator<=>(const A &, const A &) = delete;
    bool operator==(const A &, const A &) = delete;
    bool operator<(const A &, const A &) = delete;
    bool cmp = N::f<A>() < N::f<A>();
  }
}

#endif
