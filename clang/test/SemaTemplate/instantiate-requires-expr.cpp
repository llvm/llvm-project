// RUN: %clang_cc1 -std=c++2a -x c++ %s -verify -Wno-unused-value

template<typename T, typename U>
constexpr bool is_same_v = false;

template<typename T>
constexpr bool is_same_v<T, T> = true;

// We use a hack in this file to make the compiler print out the requires
// expression after it has been instantiated - we put false_v<requires {...}> as
// the requires clause of a class template, then instantiate the template.
// The requirement will not be satisfied, and the explaining diagnostic will
// print out false_v<requires {...}> in its raw form (the false_v serves to
// prevent the diagnostic from elaborating on why the requires expr wasn't
// satisfied).

template<bool v>
constexpr bool false_v = false;

template<typename... Ts>
using void_t = void;

// Check that requires parameters are instantiated correctly.

template<typename T> requires
false_v<requires (T t) { requires is_same_v<decltype(t), int>; }>
// expected-note@-1 {{because 'false_v<requires (int t) { requires is_same_v<decltype(t), int>; }>' evaluated to false}}
// expected-note@-2 {{because 'false_v<requires (char t) { requires is_same_v<decltype(t), int>; }>' evaluated to false}}
struct r1 {};

using r1i1 = r1<int>; // expected-error {{constraints not satisfied for class template 'r1' [with T = int]}}
using r1i2 = r1<char>; // expected-error {{constraints not satisfied for class template 'r1' [with T = char]}}

// Check that parameter packs work.

template<typename... Ts> requires
false_v<requires (Ts... ts) {requires ((sizeof(ts) == 2) && ...);}>
// expected-note@-1 {{because 'false_v<requires (short ts, unsigned short ts) { requires (sizeof (ts) == 2) && (sizeof (ts) == 2); }>'}}
// expected-note@-2 {{because 'false_v<requires (short ts) { requires (sizeof (ts) == 2); }>' evaluated to false}}
struct r2 {};

using r2i1 = r2<short, unsigned short>; // expected-error {{constraints not satisfied for class template 'r2' [with Ts = <short, unsigned short>]}}
using r2i2 = r2<short>; // expected-error {{constraints not satisfied for class template 'r2' [with Ts = <short>]}}

template<typename... Ts> requires
false_v<(requires (Ts ts) {requires sizeof(ts) != 0;} && ...)>
// expected-note@-1 {{because 'false_v<requires (short ts) { requires sizeof (ts) != 0; } && requires (unsigned short ts) { requires sizeof (ts) != 0; }>' evaluated to false}}
// expected-note@-2 {{because 'false_v<requires (short ts) { requires sizeof (ts) != 0; }>' evaluated to false}}
struct r3 {};

using r3i1 = r3<short, unsigned short>; // expected-error {{constraints not satisfied for class template 'r3' [with Ts = <short, unsigned short>]}}
using r3i2 = r3<short>; // expected-error {{constraints not satisfied for class template 'r3' [with Ts = <short>]}}

template<typename T>
struct identity { using type = T; };

namespace type_requirement {
  struct A {};

  // check that nested name specifier is instantiated correctly.
  template<typename T> requires false_v<requires { typename T::type; }> // expected-note{{because 'false_v<requires { typename identity<int>::type; }>' evaluated to false}}
  struct r1 {};

  using r1i = r1<identity<int>>; // expected-error{{constraints not satisfied for class template 'r1' [with T = identity<int>]}}

  // check that template argument list is instantiated correctly.
  template<typename T>
  struct contains_template {
      template<typename U> requires is_same_v<contains_template<T>, U>
      using temp = int;
  };

  template<typename T> requires
  false_v<requires { typename T::template temp<T>; }>
  // expected-note@-1 {{because 'false_v<requires { typename contains_template<int>::template temp<contains_template<int>>; }>' evaluated to false}}
  // expected-note@-2 {{because 'false_v<requires { typename contains_template<short>::template temp<contains_template<short>>; }>' evaluated to false}}
  struct r2 {};

  using r2i1 = r2<contains_template<int>>; // expected-error{{constraints not satisfied for class template 'r2' [with T = contains_template<int>]}}
  using r2i2 = r2<contains_template<short>>; // expected-error{{constraints not satisfied for class template 'r2' [with T = contains_template<short>]}}

  // substitution error occurs, then requires expr is instantiated again

  template<typename T>
  struct a {
      template<typename U> requires (requires { typename T::a::a; }, false)
      // expected-note@-1{{because 'requires { <<error-type>>; } , false' evaluated to false}}
      struct r {};
  };

  using ari = a<int>::r<short>; // expected-error{{constraints not satisfied for class template 'r' [with U = short]}}

  // Parameter pack inside expr
  template<typename... Ts> requires
  false_v<(requires { typename Ts::type; } && ...)>
  // expected-note@-1 {{because 'false_v<requires { typename identity<short>::type; } && requires { typename identity<int>::type; } && requires { <<error-type>>; }>' evaluated to false}}
  struct r5 {};

  using r5i = r5<identity<short>, identity<int>, short>; // expected-error{{constraints not satisfied for class template 'r5' [with Ts = <identity<short>, identity<int>, short>]}}
  template<typename... Ts> requires
  false_v<(requires { typename void_t<Ts>; } && ...)> // expected-note{{because 'false_v<requires { typename void_t<int>; } && requires { typename void_t<short>; }>' evaluated to false}}
  struct r6 {};

  using r6i = r6<int, short>; // expected-error{{constraints not satisfied for class template 'r6' [with Ts = <int, short>]}}

  template<typename... Ts> requires
  false_v<(requires { typename Ts::template aaa<Ts>; } && ...)>
  // expected-note@-1 {{because 'false_v<requires { <<error-type>>; } && requires { <<error-type>>; }>' evaluated to false}}
  struct r7 {};

  using r7i = r7<int, A>; // expected-error{{constraints not satisfied for class template 'r7' [with Ts = <int, A>]}}
}

namespace expr_requirement {
  // check that compound/simple requirements are instantiated correctly.

  template<typename T> requires false_v<requires { sizeof(T); { sizeof(T) }; }>
  // expected-note@-1 {{because 'false_v<requires { sizeof(int); { sizeof(int) }; }>' evaluated to false}}
  // expected-note@-2 {{because 'false_v<requires { <<error-expression>>; { sizeof(T) }; }>' evaluated to false}}
  struct r1 {};

  using r1i1 = r1<int>; // expected-error{{constraints not satisfied for class template 'r1' [with T = int]}}
  using r1i2 = r1<void>; // expected-error{{constraints not satisfied for class template 'r1' [with T = void]}}

  // substitution error occurs in expr, then expr is instantiated again.

  template<typename T>
  struct a {
      template<typename U> requires (requires { sizeof(T::a); }, false) // expected-note{{because 'requires { <<error-expression>>; } , false' evaluated to false}}
      struct r {};
  };

  using ari = a<int>::r<short>; // expected-error{{constraints not satisfied for class template 'r' [with U = short]}}

  // check that the return-type-requirement is instantiated correctly.

  template<typename T, typename U = int>
  concept C1 = is_same_v<T, U>;

  template<typename T> requires false_v<requires(T t) { { t } -> C1<T>; }>
  // expected-note@-1 {{because 'false_v<requires (int t) { { t } -> C1<int>; }>' evaluated to false}}
  // expected-note@-2 {{because 'false_v<requires (double t) { { t } -> C1<double>; }>' evaluated to false}}
  struct r2 {};

  using r2i1 = r2<int>; // expected-error{{constraints not satisfied for class template 'r2' [with T = int]}}
  using r2i2 = r2<double>; // expected-error{{constraints not satisfied for class template 'r2' [with T = double]}}


  // substitution error occurs in return type requirement, then requires expr is
  // instantiated again.

  template<typename T>
  struct b {
      template<typename U> requires (requires { { 0 } -> C1<typename T::a>; }, false) // expected-note{{because 'requires { { 0 } -> <<error-type>>; } , false' evaluated to false}}
      struct r {};
  };

  using bri = b<int>::r<short>; // expected-error{{constraints not satisfied for class template 'r' [with U = short]}}


  template<typename... Ts> requires
  false_v<(requires { { 0 } noexcept -> C1<Ts>; } && ...)>
  // expected-note@-1 {{because 'false_v<requires { { 0 } noexcept -> C1<int>; } && requires { { 0 } noexcept -> C1<unsigned int>; }>' evaluated to false}}
  struct r3 {};

  using r3i = r3<int, unsigned int>; // expected-error{{constraints not satisfied for class template 'r3' [with Ts = <int, unsigned int>]}}

  template<typename T>
  struct r4 {
      constexpr int foo() {
        if constexpr (requires { this->invalid(); })
          return 1;
        else
          return 0;
      }

      constexpr void invalid() requires false { }
  };
  static_assert(r4<int>{}.foo() == 0);
}

namespace nested_requirement {
  // check that constraint expression is instantiated correctly
  template<typename T> requires false_v<requires { requires sizeof(T) == 2; }> // expected-note{{because 'false_v<requires { requires sizeof(int) == 2; }>' evaluated to false}}
  struct r1 {};

  using r1i = r1<int>; // expected-error{{constraints not satisfied for class template 'r1' [with T = int]}}

  // substitution error occurs in expr, then expr is instantiated again.
  template<typename T>
  struct a {
      template<typename U> requires
      (requires { requires sizeof(T::a) == 0; }, false) // expected-note{{because 'requires { requires <<error-expression>>; } , false' evaluated to false}}
      struct r {};
  };

  using ari = a<int>::r<short>; // expected-error{{constraints not satisfied for class template 'r' [with U = short]}}

  // Parameter pack inside expr
  template<typename... Ts> requires
  false_v<(requires { requires sizeof(Ts) == 0; } && ...)>
  // expected-note@-1 {{because 'false_v<requires { requires sizeof(int) == 0; } && requires { requires sizeof(short) == 0; }>' evaluated to false}}
  struct r2 {};

  using r2i = r2<int, short>; // expected-error{{constraints not satisfied for class template 'r2' [with Ts = <int, short>]}}
}

// Parameter pack inside multiple requirements
template<typename... Ts> requires
false_v<(requires { requires sizeof(Ts) == 0; sizeof(Ts); } && ...)>
// expected-note@-1 {{because 'false_v<requires { requires sizeof(int) == 0; sizeof(Ts); } && requires { requires sizeof(short) == 0; sizeof(Ts); }>' evaluated to false}}
struct r4 {};

using r4i = r4<int, short>; // expected-error{{constraints not satisfied for class template 'r4' [with Ts = <int, short>]}}

template<typename... Ts> requires
false_v<(requires(Ts t) { requires sizeof(t) == 0; t++; } && ...)>
// expected-note@-1 {{because 'false_v<requires (int t) { requires sizeof (t) == 0; t++; } && requires (short t) { requires sizeof (t) == 0; t++; }>' evaluated to false}}
struct r5 {};

using r5i = r5<int, short>; // expected-error{{constraints not satisfied for class template 'r5' [with Ts = <int, short>]}}

template<typename T> requires
false_v<(requires(T t) { T{t}; })> // T{t} creates an "UnevaluatedList" context.
// expected-note@-1 {{because 'false_v<(requires (int t) { int{t}; })>' evaluated to false}}
struct r6 {};

using r6i = r6<int>;
// expected-error@-1 {{constraints not satisfied for class template 'r6' [with T = int]}}

namespace GH73885 {

template <class> // expected-error {{extraneous}}
template <class T> requires(T{})
constexpr bool e_v = true;

static_assert(e_v<bool>);

} // namespace GH73885

namespace GH84020 {

struct B {
  template <typename S> void foo();
  void bar();
};

template <typename T, typename S> struct A : T {
  void foo() {
    static_assert(requires { T::template foo<S>(); });
    static_assert(requires { T::bar(); });
  }
};

template class A<B, double>;

} // namespace GH84020

namespace GH110785 {

struct Foo {
  static void f(auto) requires(false) {}
  void f(int) {}

  static_assert([](auto v) {
    return requires { f(v); };
  } (0) == false);
};

} // namespace GH110785

namespace sugared_instantiation {
  template <class C1> concept C = requires { C1{}; };
  template <class D1> concept D = requires { new D1; };

  // Test that 'deduced auto' doesn't get confused with 'undeduced auto'.
  auto f() { return 0; }
  static_assert(requires { { f() } -> C; });
  static_assert(requires { { f() } -> D; });
} // namespace sugared_instantiation

namespace GH184562 {

template <typename T>
struct tid {
  using type = T;
};
template <typename T>
using tid_t = tid<T>::type;

template <class T, int N = tid_t<T>::value>
concept req = N < 3 || requires { typename T::template as_two<3>; };

struct two {
  static constexpr int value = 2;
};

struct three {
  static constexpr int value = 3;
  template <int>
  using as_two = two;
};

template <req T>
struct test {
  using type = test<typename T::template as_two<1>>;
};

test<three> x;

}

namespace SusbtitutionFailureInArguments1 {
  template <class T1, class T2 = typename T1::type>
    concept C = sizeof(T1) == sizeof(T2);

  template<class>
    requires requires { { 0 } -> C; }
    void f();

  // FIXME: We should explain the substitution failure in a note.
  template<class T>
    requires requires { { T() } -> C; } // expected-note {{because 'C<expr-type>' would be invalid}}
    void g();
  // expected-note@-1 {{ignored: constraints not satisfied}}
  template void g<int>();
  // expected-error@-1 {{does not refer to a function template}}

  // FIXME: static_assert should gain support for explaining non-satisfied requirements.
  static_assert(requires { { 0 } -> C; });
  // expected-error@-1 {{static assertion failed due to requirement 'requires { { <<error-expression>> } -> C; }'}}
} // namespace SubstitutionFailureInArguments1
