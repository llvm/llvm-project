// RUN: %clang_cc1 -std=c++2a -x c++ %s -verify

static_assert(requires { requires true; });

template<typename T> requires requires { requires false; } // expected-note{{because 'false' evaluated to false}}
struct r1 {};

using r1i = r1<int>; // expected-error{{constraints not satisfied for class template 'r1' [with T = int]}}

template<typename T> requires requires { requires sizeof(T) == 0; } // expected-note{{because 'sizeof(int) == 0' (4 == 0) evaluated to false}}
struct r2 {};

using r2i = r2<int>; // expected-error{{constraints not satisfied for class template 'r2' [with T = int]}}

template<typename T> requires requires (T t) { requires sizeof(t) == 0; } // expected-note{{because 'sizeof (t) == 0' (4 == 0) evaluated to false}}
struct r3 {};

using r3i = r3<int>; // expected-error{{constraints not satisfied for class template 'r3' [with T = int]}}

template<typename T>
struct X {
    template<typename U> requires requires (U u) { requires sizeof(u) == sizeof(T); } // expected-note{{because 'sizeof (u) == sizeof(T)' would be invalid: invalid application of 'sizeof' to an incomplete type 'void'}}
    struct r4 {};
};

using r4i = X<void>::r4<int>; // expected-error{{constraints not satisfied for class template 'r4' [with U = int]}}

// C++ [expr.prim.req.nested] Examples
namespace std_example {
  template<typename U> concept C1 = sizeof(U) == 1; // expected-note{{because 'sizeof(decltype(+t)) == 1' (4 == 1) evaluated to false}}
  template<typename T> concept D =
    requires (T t) {
      requires C1<decltype (+t)>; // expected-note{{because 'decltype(+t)' (aka 'int') does not satisfy 'C1'}}
  };

  struct T1 { char operator+() { return 'a'; } };
  static_assert(D<T1>);
  template<D T> struct D_check {}; // expected-note{{because 'short' does not satisfy 'D'}}
  using dc1 = D_check<short>; // expected-error{{constraints not satisfied for class template 'D_check' [with T = short]}}

  template<typename T>
  concept C2 = requires (T a) {
      requires sizeof(a) == 4; // OK
      requires a == 0; // expected-note{{because 'a == 0' would be invalid: constraint variable 'a' cannot be used in an evaluated context}}
    };
  static_assert(C2<int>); // expected-note{{because 'int' does not satisfy 'C2'}} expected-error{{static assertion failed}}
}

template<typename T>
concept K = requires (T::Type X) {
  X.next();
};

