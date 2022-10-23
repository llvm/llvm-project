// RUN: %clang_cc1 -std=c++11 -verify %s -pedantic
// RUN: %clang_cc1 -std=c++20 -verify %s -pedantic


namespace PR31692 {
  struct A {
    struct X { int n = 0; } x;
    // Trigger construction of X() from a SFINAE context. This must not mark
    // any part of X as invalid.
    static_assert(!__is_constructible(X), "");
    // Check that X::n is not marked invalid.
    double &r = x.n; // expected-error {{non-const lvalue reference to type 'double' cannot bind to a value of unrelated type 'int'}}
  };
  // A::X can now be default-constructed.
  static_assert(__is_constructible(A::X), "");
}


struct S {
} constexpr s;
struct C {
  C(S);
};
class MemInit {
  C m = s;
};

#if __cplusplus >= 202002L
// This test ensures cleanup expressions are correctly produced
// in the presence of default member initializers.
namespace PR136554 {
struct string {
  constexpr string(const char*) {};
  constexpr ~string();
};
struct S;
struct optional {
    template <typename U = S>
    constexpr optional(U &&) {}
};
struct S {
    string a;
    optional b;
    int defaulted = 0;
} test {
    "", {
        { "", 0 }
    }
};
}
#endif
