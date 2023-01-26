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
  template<typename U> concept C1 = sizeof(U) == 1; // expected-note{{because 'sizeof(int) == 1' (4 == 1) evaluated to false}}
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

namespace SubstitutionFailureNestedRequires {
template<class T>  concept True = true;
template<class T>  concept False = false;

struct S { double value; };

template <class T>
concept Pipes = requires (T x) {
   requires True<decltype(x.value)> || True<T> || False<T>;
   requires False<T> || True<T> || True<decltype(x.value)>;
};

template <class T>
concept Amps1 = requires (T x) {
   requires True<decltype(x.value)> && True<T> && !False<T>; // #Amps1
};
template <class T>
concept Amps2 = requires (T x) {
   requires True<T> && True<decltype(x.value)>;
};

static_assert(Pipes<S>);
static_assert(Pipes<double>);

static_assert(Amps1<S>);
static_assert(!Amps1<double>);

static_assert(Amps2<S>);
static_assert(!Amps2<double>);

template<class T>
void foo1() requires requires (T x) { // #foo1
  requires
  True<decltype(x.value)> // #foo1Value
  && True<T>;
} {}
template<class T> void fooPipes() requires Pipes<T> {}
template<class T> void fooAmps1() requires Amps1<T> {} // #fooAmps1
void foo() {
  foo1<S>();
  foo1<int>(); // expected-error {{no matching function for call to 'foo1'}}
  // expected-note@#foo1Value {{because 'True<decltype(x.value)> && True<T>' would be invalid: member reference base type 'int' is not a structure or union}}
  // expected-note@#foo1 {{candidate template ignored: constraints not satisfied [with T = int]}}
  fooPipes<S>();
  fooPipes<int>();
  fooAmps1<S>();
  fooAmps1<int>(); // expected-error {{no matching function for call to 'fooAmps1'}}
  // expected-note@#fooAmps1 {{candidate template ignored: constraints not satisfied [with T = int]}}
  // expected-note@#fooAmps1 {{because 'int' does not satisfy 'Amps1'}}
  // expected-note@#Amps1 {{because 'True<decltype(x.value)> && True<T> && !False<T>' would be invalid: member reference base type 'int' is not a structure or union}}
}

template<class T>
concept HasNoValue = requires (T x) {
  requires !True<decltype(x.value)> && True<T>;
};
// FIXME: 'int' does not satisfy 'HasNoValue' currently since `!True<decltype(x.value)>` is an invalid expression.
// But, in principle, it should be constant-evaluated to true.
// This happens also for requires expression and is not restricted to nested requirement.
static_assert(!HasNoValue<int>);
static_assert(!HasNoValue<S>);

template<class T> constexpr bool NotAConceptTrue = true;
template <class T>
concept SFinNestedRequires = requires (T x) {
    // SF in a non-concept specialisation should also be evaluated to false.
   requires NotAConceptTrue<decltype(x.value)> || NotAConceptTrue<T>;
};
static_assert(SFinNestedRequires<int>);
static_assert(SFinNestedRequires<S>);
template <class T>
void foo() requires SFinNestedRequires<T> {}
void bar() {
  foo<int>();
  foo<S>();
}
namespace ErrorExpressions_NotSF {
template<typename T> struct X { static constexpr bool value = T::value; }; // #X_Value
struct True { static constexpr bool value = true; };
struct False { static constexpr bool value = false; };
template<typename T> concept C = true;
template<typename T> concept F = false;

template<typename T> requires requires(T) { requires C<T> || X<T>::value; } void foo();

template<typename T> requires requires(T) { requires C<T> && X<T>::value; } void bar(); // #bar
template<typename T> requires requires(T) { requires F<T> || (X<T>::value && C<T>); } void baz();

void func() {
  foo<True>();
  foo<False>();
  foo<int>();

  bar<True>();
  bar<False>();
  // expected-error@-1 {{no matching function for call to 'bar'}}
  // expected-note@#bar {{while substituting template arguments into constraint expression here}}
  // expected-note@#bar {{while checking the satisfaction of nested requirement requested here}}
  // expected-note@#bar {{candidate template ignored: constraints not satisfied [with T = False]}}
  // expected-note@#bar {{because 'X<False>::value' evaluated to false}}

  bar<int>();
  // expected-note@-1 {{while checking constraint satisfaction for template 'bar<int>' required here}} \
  // expected-note@-1 {{in instantiation of function template specialization}}
  // expected-note@#bar {{in instantiation of static data member}}
  // expected-note@#bar {{in instantiation of requirement here}}
  // expected-note@#bar {{while checking the satisfaction of nested requirement requested here}}
  // expected-note@#bar {{while substituting template arguments into constraint expression here}}
  // expected-error@#X_Value {{type 'int' cannot be used prior to '::' because it has no members}}
}
}
}

namespace no_crash_D138914 {
// https://reviews.llvm.org/D138914
template <class a, a> struct b;
template <bool c> using d = b<bool, c>;
template <class a, class e> using f = d<__is_same(a, e)>;
template <class a, class e>
concept g = f<a, e>::h;
template <class a, class e>
concept i = g<e, a>;
template <typename> class j { // expected-note {{candidate template ignored}}
  template <typename k>
  requires requires { requires i<j, k>; }
  j(); // expected-note {{candidate template ignored}}
};
template <> j(); // expected-error {{deduction guide declaration without trailing return type}} // expected-error {{no function template}}
}
