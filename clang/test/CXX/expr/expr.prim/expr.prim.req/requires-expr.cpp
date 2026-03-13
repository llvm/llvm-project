// RUN: %clang_cc1 -std=c++2a -x c++ %s -verify

using A = int;

template<typename T, typename U>
constexpr bool is_same_v = false;

template<typename T>
constexpr bool is_same_v<T, T> = true;

template<typename T, typename U>
concept same_as = is_same_v<T, U>;

static_assert(requires { requires true; 0; typename A;
                         { 0 } -> same_as<int>; });
static_assert(is_same_v<bool, decltype(requires { requires false; })>);

// Check that requires expr is an unevaluated context.
struct Y {
  int i;
  static constexpr bool r = requires { i; };
};

template<typename T> requires requires (T t) {
  requires false; // expected-note{{because 'false' evaluated to false}}
  requires false;
  requires requires {
    requires false;
  };
}
struct r1 { };

using r1i = r1<int>;
// expected-error@-1 {{constraints not satisfied for class template 'r1' [with T = int]}}

template<typename T> requires requires (T t) {
  requires requires {
    requires false; // expected-note{{because 'false' evaluated to false}}
  };
}
struct r2 { };

using r2i = r2<int>;
// expected-error@-1 {{constraints not satisfied for class template 'r2' [with T = int]}}

template<typename T> requires requires (T t) {
  requires requires {
    requires true;
  };
  requires true;
  requires requires {
    requires false; // expected-note{{because 'false' evaluated to false}}
  };
}
struct r3 { };

using r3i = r3<int>;
// expected-error@-1 {{constraints not satisfied for class template 'r3' [with T = int]}}

template<typename T>
struct S { static const int s = T::value; };

template<typename T> requires requires { T::value; S<T>::s; }
// expected-note@-1 {{because 'T::value' would be invalid: type 'int' cannot be used prior to '::' because it has no members}}
struct r4 { };

using r4i = r4<int>;
// expected-error@-1 {{constraints not satisfied for class template 'r4' [with T = int]}}

namespace GH109538 {
static_assert(requires(void *t) { t; });
static_assert(requires(void) { 42; });
static_assert(requires(void t) { // expected-error {{argument may not have 'void' type}}
  t;
});
static_assert(requires(void t, int a) {  // expected-error {{'void' must be the first and only parameter if specified}}
  t;
});
static_assert(requires(const void) { // expected-error {{'void' as parameter must not have type qualifiers}}
  42;
});
} // namespace GH109538
