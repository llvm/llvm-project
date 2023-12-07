// RUN: %clang_cc1 -std=c++2a -x c++ %s -verify

static_assert(requires { { 0 }; });
static_assert(requires { { "aaaa" }; });
static_assert(requires { { (0).da }; }); // expected-error{{member reference base type 'int' is not a structure or union}}

void foo() {}
static_assert(requires { { foo() }; });

// Substitution failure in expression

struct A {};
struct B {
    B operator+(const B &other) const { return other; }
};
struct C {
    C operator+(C &other) const { return other; }
};

template<typename T> requires requires (T a, const T& b) { { a + b }; } // expected-note{{because 'a + b' would be invalid: invalid operands to binary expression ('A' and 'const A')}} expected-note{{because 'a + b' would be invalid: invalid operands to binary expression ('C' and 'const C')}}
struct r1 {};

using r1i1 = r1<int>;
using r1i2 = r1<A>; // expected-error{{constraints not satisfied for class template 'r1' [with T = A]}}
using r1i3 = r1<B>;
using r1i4 = r1<C>; // expected-error{{constraints not satisfied for class template 'r1' [with T = C]}}

struct D { void foo() {} };

template<typename T> requires requires (T a) { { a.foo() }; } // expected-note{{because 'a.foo()' would be invalid: no member named 'foo' in 'A'}} expected-note{{because 'a.foo()' would be invalid: member reference base type 'int' is not a structure or union}} expected-note{{because 'a.foo()' would be invalid: 'this' argument to member function 'foo' has type 'const D', but function is not marked const}}
struct r2 {};

using r2i1 = r2<int>; // expected-error{{constraints not satisfied for class template 'r2' [with T = int]}}
using r2i2 = r2<A>; // expected-error{{constraints not satisfied for class template 'r2' [with T = A]}}
using r2i3 = r2<D>;
using r2i4 = r2<const D>; // expected-error{{constraints not satisfied for class template 'r2' [with T = const D]}}

template<typename T> requires requires { { sizeof(T) }; } // expected-note{{because 'sizeof(T)' would be invalid: invalid application of 'sizeof' to an incomplete type 'void'}} expected-note{{because 'sizeof(T)' would be invalid: invalid application of 'sizeof' to an incomplete type 'nonexistent'}}
struct r3 {};

using r3i1 = r3<int>;
using r3i2 = r3<A>;
using r3i3 = r3<A &>;
using r3i4 = r3<void>; // expected-error{{constraints not satisfied for class template 'r3' [with T = void]}}
using r3i4 = r3<class nonexistent>; // expected-error{{constraints not satisfied for class template 'r3' [with T = nonexistent]}}

// Non-dependent expressions

template<typename T> requires requires (T t) { { 0 }; { "a" }; { (void)'a' }; }
struct r4 {};

using r4i1 = r4<int>;
using r4i2 = r4<int[10]>;
using r4i3 = r4<int(int)>;

// Noexcept requirement
void maythrow() { }
static_assert(!requires { { maythrow() } noexcept; });
static_assert(requires { { 1 } noexcept; });

struct E { void operator++(int) noexcept; };
struct F { void operator++(int); };

template<typename T> requires requires (T t) { { t++ } noexcept; } // expected-note{{because 't ++' may throw an exception}}
struct r5 {};

using r5i1 = r5<int>;
using r5i2 = r5<E>;
using r5i2 = r5<F>; // expected-error{{constraints not satisfied for class template 'r5' [with T = F]}}

template<typename T> requires requires (T t) { { t.foo() } noexcept; } // expected-note{{because 't.foo()' would be invalid: no member named 'foo' in 'E'}}
struct r6 {};

using r6i = r6<E>; // expected-error{{constraints not satisfied for class template 'r6' [with T = E]}}

template<typename T, typename U>
constexpr bool is_same_v = false;

template<typename T>
constexpr bool is_same_v<T, T> = true;

template<typename T> struct remove_reference { using type = T; };
template<typename T> struct remove_reference<T&> { using type = T; };

template<typename T, typename U>
concept Same = is_same_v<T, U>;

template<typename T>
concept Large = sizeof(typename remove_reference<T>::type) >= 4;
// expected-note@-1{{because 'sizeof(typename remove_reference<short &>::type) >= 4' (2 >= 4) evaluated to false}}

template<typename T> requires requires (T t) { { t } -> Large; } // expected-note{{because 'short &' does not satisfy 'Large':}}
struct r7 {};

using r7i1 = r7<int>;
using r7i2 = r7<short>; // expected-error{{constraints not satisfied for class template 'r7' [with T = short]}}

template<typename T> requires requires (T t) { { t } -> Same<T&>; }
struct r8 {};

using r8i1 = r8<int>;
using r8i2 = r8<short*>;

// Substitution failure in type constraint

template<typename T> requires requires (T t) { { t } -> Same<typename T::type&>; }
// expected-note@-1{{because 'Same<expr-type, typename T::type &>' would be invalid: type 'int' cannot be used prior to '::' because it has no members}}
struct r9 {};

struct M { using type = M; };

using r9i1 = r9<M>;
using r9i2 = r9<int>; // expected-error{{constraints not satisfied for class template 'r9' [with T = int]}}

// Substitution failure in both expression and return type requirement

template<typename T> requires requires (T t) { { t.foo() } -> Same<typename T::type>; } // expected-note{{because 't.foo()' would be invalid: member reference base type 'int' is not a structure or union}}
struct r10 {};

using r10i = r10<int>; // expected-error{{constraints not satisfied for class template 'r10' [with T = int]}}

// Non-type concept in type constraint

template<int T>
concept IsEven = (T % 2) == 0;

template<typename T> requires requires (T t) { { t } -> IsEven; } // expected-error{{concept named in type constraint is not a type concept}}
struct r11 {};

// Value categories

template<auto a = 0>
requires requires (int b) {
  { a } -> Same<int>;
  { b } -> Same<int&>;
  { 0 } -> Same<int>;
  { static_cast<int&&>(a) } -> Same<int&&>;
} void f1() {}
template void f1<>();

// C++ [expr.prim.req.compound] Example
namespace std_example {
  template<typename T> concept C1 =
    requires(T x) {
      {x++};
    };

  template<typename T, typename U> constexpr bool is_same_v = false;
  template<typename T> constexpr bool is_same_v<T, T> = true;

  template<typename T, typename U> concept same_as = is_same_v<T, U>;
  // expected-note@-1 {{because 'is_same_v<int, int *>' evaluated to false}}

  static_assert(C1<int>);
  static_assert(C1<int*>);
  template<C1 T> struct C1_check {};
  using c1c1 = C1_check<int>;
  using c1c2 = C1_check<int[10]>;

  template<typename T> concept C2 =
    requires(T x) {
      {*x} -> same_as<typename T::inner>;
      // expected-note@-1{{because type constraint 'same_as<int, typename T2::inner>' was not satisfied:}}
      // expected-note@-2{{because '*x' would be invalid: indirection requires pointer operand ('int' invalid)}}
    };

  struct T1 {
    using inner = int;
    inner operator *() { return 0; }
  };
  struct T2 {
    using inner = int *;
    int operator *() { return 0; }
  };
  static_assert(C2<T1>);
  template<C2 T> struct C2_check {}; // expected-note{{because 'int' does not satisfy 'C2'}} expected-note{{because 'std_example::T2' does not satisfy 'C2'}}
  using c2c1 = C2_check<int>; // expected-error{{constraints not satisfied for class template 'C2_check' [with T = int]}}
  using c2c2 = C2_check<T2>; // expected-error{{constraints not satisfied for class template 'C2_check' [with T = std_example::T2]}}

  template<typename T>
  void g(T t) noexcept(sizeof(T) == 1) {}

  template<typename T> concept C5 =
    requires(T x) {
      {g(x)} noexcept; // expected-note{{because 'g(x)' may throw an exception}}
    };

  static_assert(C5<char>);
  template<C5 T> struct C5_check {}; // expected-note{{because 'short' does not satisfy 'C5'}}
  using c5 = C5_check<short>; // expected-error{{constraints not satisfied for class template 'C5_check' [with T = short]}}
}
