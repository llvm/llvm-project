// RUN: %clang -fsyntax-only -std=c++2a -Xclang -verify -ftemplate-depth=5 -ftemplate-backtrace-limit=4 %s

// RequiresExpr contains invalid requirement. (Eg. Highly recursive template).
template<int x>
struct A { static constexpr bool far(); };
class B {
    bool data_member;
    friend struct A<1>;
};

template<>
constexpr bool A<0>::far() { return true; }

template<int x>
constexpr bool A<x>::far() {
    return requires(B b) {
      b.data_member;
      requires A<x-1>::far(); // #Invalid
      // expected-error@#Invalid {{recursive template instantiation exceeded maximum depth}}
      // expected-note@#Invalid 3 {{while}}
      // expected-note@#Invalid {{contexts in backtrace}}
      // expected-note@#Invalid {{use -ftemplate-depth=N to increase}}
    };
}
static_assert(A<1>::far());
static_assert(!A<6>::far()); // expected-note {{in instantiation of member function}}
