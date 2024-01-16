// RUN: %clang_cc1 -std=c++1z -verify %s -DERRORS -Wundefined-func-template
// RUN: %clang_cc1 -std=c++1z -verify %s -UERRORS -Wundefined-func-template

// This test is split into two because we only produce "undefined internal"
// warnings if we didn't produce any errors.
#if ERRORS

namespace std {
  using size_t = decltype(sizeof(0));
  template<typename T> struct initializer_list {
    const T *p;
    size_t n;
    initializer_list();
  };
  // FIXME: This should probably not be necessary.
  template<typename T> initializer_list(initializer_list<T>) -> initializer_list<T>;
}

template<typename T> constexpr bool has_type(...) { return false; }
template<typename T> constexpr bool has_type(T&) { return true; }

std::initializer_list il = {1, 2, 3, 4, 5};

template<typename T> struct vector {
  template<typename Iter> vector(Iter, Iter);
  vector(std::initializer_list<T>);
};

template<typename T> vector(std::initializer_list<T>) -> vector<T>;
template<typename Iter> explicit vector(Iter, Iter) -> vector<typename Iter::value_type>;
template<typename T> explicit vector(std::size_t, T) -> vector<T>;

vector v1 = {1, 2, 3, 4};
static_assert(has_type<vector<int>>(v1));

struct iter { typedef char value_type; } it, end;
vector v2(it, end);
static_assert(has_type<vector<char>>(v2));

vector v3(5, 5);
static_assert(has_type<vector<int>>(v3));

vector v4 = {it, end};
static_assert(has_type<vector<iter>>(v4));

vector v5{it, end};
static_assert(has_type<vector<iter>>(v5));

template<typename ...T> struct tuple { tuple(T...); };
template<typename ...T> explicit tuple(T ...t) -> tuple<T...>; // expected-note {{declared}}
// FIXME: Remove
template<typename ...T> tuple(tuple<T...>) -> tuple<T...>;

const int n = 4;
tuple ta = tuple{1, 'a', "foo", n};
static_assert(has_type<tuple<int, char, const char*, int>>(ta));

tuple tb{ta};
static_assert(has_type<tuple<int, char, const char*, int>>(tb));

// FIXME: This should be tuple<tuple<...>>; when the above guide is removed.
tuple tc = {ta};
static_assert(has_type<tuple<int, char, const char*, int>>(tc));

tuple td = {1, 2, 3}; // expected-error {{selected an explicit deduction guide}}
static_assert(has_type<tuple<int, char, const char*, int>>(td));

// FIXME: This is a GCC extension for now; if CWG don't allow this, at least
// add a warning for it.
namespace new_expr {
  tuple<int> *p = new tuple{0};
  tuple<float, float> *q = new tuple(1.0f, 2.0f);
}

namespace ambiguity {
  template<typename T> struct A {};
  A(unsigned short) -> A<int>; // expected-note {{candidate}}
  A(short) -> A<int>; // expected-note {{candidate}}
  A a = 0; // expected-error {{ambiguous deduction for template arguments of 'A'}}

  template<typename T> struct B {};
  template<typename T> B(T(&)(int)) -> B<int>; // expected-note {{candidate function [with T = int]}}
  template<typename T> B(int(&)(T)) -> B<int>; // expected-note {{candidate function [with T = int]}}
  int f(int);
  B b = f; // expected-error {{ambiguous deduction for template arguments of 'B'}}
}

// FIXME: Revisit this once CWG decides if attributes, and [[deprecated]] in
// particular, should be permitted here.
namespace deprecated {
  template<typename T> struct A { A(int); };
  [[deprecated]] A(int) -> A<void>; // expected-note {{marked deprecated here}}
  A a = 0; // expected-warning {{'<deduction guide for A>' is deprecated}}
}

namespace dependent {
  template<template<typename...> typename A> decltype(auto) a = A{1, 2, 3};
  static_assert(has_type<vector<int>>(a<vector>));
  static_assert(has_type<tuple<int, int, int>>(a<tuple>));

  struct B {
    template<typename T> struct X { X(T); };
    X(int) -> X<int>;
    template<typename T> using Y = X<T>; // expected-note {{template}}
  };
  template<typename T> void f() {
    typename T::X tx = 0;
    typename T::Y ty = 0; // expected-error {{alias template 'Y' requires template arguments; argument deduction only allowed for class templates}}
  }
  template void f<B>(); // expected-note {{in instantiation of}}

  template<typename T> struct C { C(T); };
  template<typename T> C(T) -> C<T>;
  template<typename T> void g(T a) {
    C b = 0;
    C c = a;
    using U = decltype(b); // expected-note {{previous}}
    using U = decltype(c); // expected-error {{different types ('C<const char *>' vs 'C<int>')}}
  }
  void h() {
    g(0);
    g("foo"); // expected-note {{instantiation of}}
  }
}

namespace look_into_current_instantiation {
  template<typename U> struct Q {};
  template<typename T> struct A {
    using U = T;
    template<typename> using V = Q<A<T>::U>;
    template<typename W = int> A(V<W>);
  };
  A a = Q<float>(); // ok, can look through class-scope typedefs and alias
                    // templates, and members of the current instantiation
  A<float> &r = a;

  template<typename T> struct B { // expected-note {{could not match 'B<T>' against 'int'}}
    struct X {
      typedef T type;
    };
    B(typename X::type); // expected-note {{couldn't infer template argument 'T'}}
  };
  B b = 0; // expected-error {{no viable}}

  // We should have a substitution failure in the immediate context of
  // deduction when using the C(T, U) constructor (probably; core wording
  // unclear).
  template<typename T> struct C {
    using U = typename T::type;
    C(T, U);
  };

  struct R { R(int); typedef R type; };
  C(...) -> C<R>;

  C c = {1, 2};
}

namespace nondeducible {
  template<typename A, typename B> struct X {};

  template<typename A> // expected-note {{non-deducible template parameter 'A'}}
  X() -> X<A, int>; // expected-error {{deduction guide template contains a template parameter that cannot be deduced}}

  template<typename A> // expected-note {{non-deducible template parameter 'A'}}
  X(typename X<A, int>::type) -> X<A, int>; // expected-error {{deduction guide template contains a template parameter that cannot be deduced}}

  template<typename A = int,
           typename B> // expected-note {{non-deducible template parameter 'B'}}
  X(int) -> X<A, B>; // expected-error {{deduction guide template contains a template parameter that cannot be deduced}}

  template<typename A = int,
           typename ...B>
  X(float) -> X<A, B...>; // ok

  template <typename> struct UnnamedTemplateParam {};
  template <typename>                                  // expected-note {{non-deducible template parameter (anonymous)}}
  UnnamedTemplateParam() -> UnnamedTemplateParam<int>; // expected-error {{deduction guide template contains a template parameter that cannot be deduced}}
}

namespace default_args_from_ctor {
  template <class A> struct S { S(A = 0) {} };
  S s(0);

  template <class A> struct T { template<typename B> T(A = 0, B = 0) {} };
  T t(0, 0);
}

namespace transform_params {
  template<typename T, T N, template<T (*v)[N]> typename U, T (*X)[N]>
  struct A {
    template<typename V, V M, V (*Y)[M], template<V (*v)[M]> typename W>
    A(U<X>, W<Y>);

    static constexpr T v = N;
  };

  int n[12];
  template<int (*)[12]> struct Q {};
  Q<&n> qn;
  A a(qn, qn);
  static_assert(a.v == 12);

  template<typename ...T> struct B {
    template<T ...V> B(const T (&...p)[V]) {
      constexpr int Vs[] = {V...};
      static_assert(Vs[0] == 3 && Vs[1] == 4 && Vs[2] == 4);
    }
    static constexpr int (*p)(T...) = (int(*)(int, char, char))nullptr;
  };
  B b({1, 2, 3}, "foo", {'x', 'y', 'z', 'w'}); // ok

  template<typename ...T> struct C {
    template<T ...V, template<T...> typename X>
      C(X<V...>);
  };
  template<int...> struct Y {};
  C c(Y<0, 1, 2>{});

  template<typename ...T> struct D {
    template<T ...V> D(Y<V...>);
  };
  D d(Y<0, 1, 2>{});
}

namespace variadic {
  int arr3[3], arr4[4];

  // PR32673
  template<typename T> struct A {
    template<typename ...U> A(T, U...);
  };
  A a(1, 2, 3);

  template<typename T> struct B {
    template<int ...N> B(T, int (&...r)[N]);
  };
  B b(1, arr3, arr4);

  template<typename T> struct C {
    template<template<typename> typename ...U> C(T, U<int>...);
  };
  C c(1, a, b);

  template<typename ...U> struct X {
    template<typename T> X(T, U...);
  };
  X x(1, 2, 3);

  template<int ...N> struct Y {
    template<typename T> Y(T, int (&...r)[N]);
  };
  Y y(1, arr3, arr4);

  template<template<typename> typename ...U> struct Z {
    template<typename T> Z(T, U<int>...);
  };
  Z z(1, a, b);
}

namespace tuple_tests {
  // The converting n-ary constructor appears viable, deducing T as an empty
  // pack (until we check its SFINAE constraints).
  namespace libcxx_1 {
    template<class ...T> struct tuple {
      template<class ...Args> struct X { static const bool value = false; };
      template<class ...U, bool Y = X<U...>::value> tuple(U &&...u);
    };
    tuple a = {1, 2, 3};
  }

  // Don't get caught by surprise when X<...> doesn't even exist in the
  // selected specialization!
  namespace libcxx_2 {
    template<class ...T> struct tuple {
      template<class ...Args> struct X { static const bool value = false; };
      // Substitution into X<U...>::value succeeds but produces the
      // value-dependent expression
      //   tuple<T...>::X<>::value
      // FIXME: Is that the right behavior?
      template<class ...U, bool Y = X<U...>::value> tuple(U &&...u);
    };
    template <> class tuple<> {};
    tuple a = {1, 2, 3}; // expected-error {{excess elements in struct initializer}}
  }

  namespace libcxx_3 {
    template<typename ...T> struct scoped_lock {
      scoped_lock(T...);
    };
    template<> struct scoped_lock<> {};
    scoped_lock l = {};
  }
}

namespace dependent {
  template<typename T> struct X { // expected-note 3{{here}}
    X(T);
  };
  template<typename T> int Var(T t) {
    X x(t);
    return X(x) + 1; // expected-error {{invalid operands}}
  }
  template<typename T> int Cast(T t) {
    return X(X(t)) + 1; // expected-error {{invalid operands}}
  }
  template<typename T> int Cast2(T t) {
    return (X)(X)t + 1; // expected-error {{deduction not allowed}}
  }
  template<typename T> int Cast3(T t) {
    return X{X{t}} + 1; // expected-error {{invalid operands}}
  }
  template<typename T> int Cast4(T t) {
    return (X){(X){t}} + 1; // expected-error 2{{deduction not allowed}}
  }
  template<typename T> int New(T t) {
    return X(new X(t)) + 1; // expected-error {{invalid operands}}
  };
  template<typename T> int *New2(T t) {
    return new X(X(t)) * 2; // expected-error {{invalid operands}}
  };
  template int Var(float); // expected-note {{instantiation of}}
  template int Cast(float); // expected-note {{instantiation of}}
  template int Cast3(float); // expected-note {{instantiation of}}
  template int New(float); // expected-note {{instantiation of}}
  template int *New2(float); // expected-note {{instantiation of}}
  template<typename T> int operator+(X<T>, int);
  template int Var(int);
  template int Cast(int);
  template int New(int);

  template<template<typename> typename Y> void test() {
    Y(0);
    new Y(0);
    Y y(0);
  }
  template void test<X>();
}

namespace injected_class_name {
  template<typename T = void> struct A {
    A();
    template<typename U> A(A<U>);
  };
  A<int> a;
  A b = a;
  using T = decltype(a);
  using T = decltype(b);
}

namespace member_guides {
  // PR34520
  template<class>
  struct Foo {
    template <class T> struct Bar {
      Bar(...) {}
    };
    Bar(int) -> Bar<int>;
  };
  Foo<int>::Bar b = 0;

  struct A {
    template<typename T> struct Public; // expected-note {{declared public}}
    Public(float) -> Public<float>;
  protected: // expected-note {{declared protected by intervening access specifier}}
    template<typename T> struct Protected; // expected-note 2{{declared protected}}
    Protected(float) -> Protected<float>;
    Public(int) -> Public<int>; // expected-error {{different access}}
  private: // expected-note {{declared private by intervening access specifier}}
    template<typename T> struct Private; // expected-note {{declared private}}
    Protected(int) -> Protected<int>; // expected-error {{different access}}
  public: // expected-note 2{{declared public by intervening access specifier}}
    template<typename T> Public(T) -> Public<T>;
    template<typename T> Protected(T) -> Protected<T>; // expected-error {{different access}}
    template<typename T> Private(T) -> Private<T>; // expected-error {{different access}}
  };
}

namespace rdar41903969 {
template <class T> struct A {};
template <class T> struct B;
template <class T> struct C {
  C(A<T>&);
  C(B<T>&);
};

void foo(A<int> &a, B<int> &b) {
  (void)C{b};
  (void)C{a};
}

template<typename T> struct X {
  X(std::initializer_list<T>) = delete;
  X(const X&);
};

template <class T> struct D : X<T> {};

void bar(D<int>& d) {
  (void)X{d};
}
}

namespace rdar41330135 {
template <int> struct A {};
template <class T>
struct S {
  template <class U>
  S(T a, U t, A<sizeof(t)>);
};
template <class T> struct D {
  D(T t, A<sizeof(t)>);
};
int f() {
  S s(0, 0, A<sizeof(int)>());
  D d(0, A<sizeof(int)>());
}

namespace test_dupls {
template<unsigned long> struct X {};
template<typename T> struct A {
  A(T t, X<sizeof(t)>);
};
A a(0, {});
template<typename U> struct B {
  B(U u, X<sizeof(u)>);
};
B b(0, {});
}

}

namespace no_crash_on_default_arg {
class A {
  template <typename T> class B {
    B(int c = 1);
  };
  // This used to crash due to unparsed default arg above. The diagnostic could
  // be improved, but the point of this test is to simply check we do not crash.
  B(); // expected-error {{deduction guide declaration without trailing return type}}
};
} // namespace no_crash_on_default_arg

#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wctad-maybe-unsupported"
namespace test_implicit_ctad_warning {

template <class T>
struct Tag {};

template <class T>
struct NoExplicit { // expected-note {{add a deduction guide to suppress this warning}}
  NoExplicit(T) {}
  NoExplicit(T, int) {}
};

// expected-warning@+1 {{'NoExplicit' may not intend to support class template argument deduction}}
NoExplicit ne(42);

template <class U>
struct HasExplicit {
  HasExplicit(U) {}
  HasExplicit(U, int) {}
};
template <class U> HasExplicit(U, int) -> HasExplicit<Tag<U>>;

HasExplicit he(42);

// Motivating examples from (taken from Stephan Lavavej's 2018 Cppcon talk)
template <class T, class U>
struct AmateurPair { // expected-note {{add a deduction guide to suppress this warning}}
  T first;
  U second;
  explicit AmateurPair(const T &t, const U &u) {}
};
// expected-warning@+1 {{'AmateurPair' may not intend to support class template argument deduction}}
AmateurPair p1(42, "hello world"); // deduces to Pair<int, char[12]>

template <class T, class U>
struct AmateurPair2 { // expected-note {{add a deduction guide to suppress this warning}}
  T first;
  U second;
  explicit AmateurPair2(T t, U u) {}
};
// expected-warning@+1 {{'AmateurPair2' may not intend to support class template argument deduction}}
AmateurPair2 p2(42, "hello world"); // deduces to Pair2<int, const char*>

template <class T, class U>
struct ProPair {
  T first; U second;
    explicit ProPair(T const& t, U  const& u)  {}
};
template<class T1, class T2>
ProPair(T1, T2) -> ProPair<T1, T2>;
ProPair p3(42, "hello world"); // deduces to ProPair<int, const char*>
static_assert(__is_same(decltype(p3), ProPair<int, const char*>));

// Test that user-defined explicit guides suppress the warning even if they
// aren't used as candidates.
template <class T>
struct TestExplicitCtor {
  TestExplicitCtor(T) {}
};
template <class T>
explicit TestExplicitCtor(TestExplicitCtor<T> const&) -> TestExplicitCtor<void>;
TestExplicitCtor<int> ce1{42};
TestExplicitCtor ce2 = ce1;
static_assert(__is_same(decltype(ce2), TestExplicitCtor<int>), "");

struct allow_ctad_t {
  allow_ctad_t() = delete;
};

template <class T>
struct TestSuppression {
  TestSuppression(T) {}
};
TestSuppression(allow_ctad_t)->TestSuppression<void>;
TestSuppression ta("abc");
static_assert(__is_same(decltype(ta), TestSuppression<const char *>), "");
}
#pragma clang diagnostic pop

namespace PR41549 {

template <class H, class P> struct umm;

template <class H = int, class P = int>
struct umm {
  umm(H h = 0, P p = 0);
};

template <class H, class P> struct umm;

umm m(1);

}

namespace PR45124 {
  class a { int d; };
  class b : a {};

  struct x { ~x(); };
  template<typename> class y { y(x = x()); };
  template<typename z> y(z)->y<z>;

  // Not a constant initializer, but trivial default initialization. We won't
  // detect this as trivial default initialization if synthesizing the implicit
  // deduction guide 'template<typename T> y(x = x()) -> Y<T>;' leaves behind a
  // pending cleanup.
  __thread b g;
}

namespace PR47175 {
  template<typename T> struct A { A(T); T x; };
  template<typename T> int &&n = A(T()).x;
  int m = n<int>;
}

// Ensure we don't crash when CTAD fails.
template <typename T1, typename T2>
struct Foo {   // expected-note{{candidate function template not viable}}
  Foo(T1, T2); // expected-note{{candidate function template not viable}}
};

template <typename... Args>
void insert(Args &&...args);

void foo() {
  insert(Foo(2, 2, 2)); // expected-error{{no viable constructor or deduction guide}}
}

namespace PR52139 {
  struct Abstract {
    template <class... Ts>
    struct overloaded : Ts... {
      using Ts::operator()...;
    };
    template <class... Ts>
    overloaded(Ts...) -> overloaded<Ts...>;

  private:
    virtual void f() = 0;
  };
}

namespace function_prototypes {
  template<class T> using fptr1 = void (*) (T);
  template<class T> using fptr2 = fptr1<fptr1<T>>;

  template<class T> void foo0(fptr1<T>) {
    static_assert(__is_same(T, const char*));
  }
  void bar0(const char *const volatile __restrict);
  void t0() { foo0(&bar0); }

  template<class T> void foo1(fptr1<const T *>) {
     static_assert(__is_same(T, char));  
  }
  void bar1(const char * __restrict);
  void t1() { foo1(&bar1); }

  template<class T> void foo2(fptr2<const T *>) {
    static_assert(__is_same(T, char));
  }
  void bar2(fptr1<const char * __restrict>);
  void t2() { foo2(&bar2); }

  template<class T> void foo3(fptr1<const T *>) {}
  void bar3(char * __restrict);
  void t3() { foo3(&bar3); }
  // expected-error@-1 {{no matching function for call to 'foo3'}}
  // expected-note@-4  {{candidate template ignored: cannot deduce a type for 'T' that would make 'const T' equal 'char'}}

  template<class T> void foo4(fptr2<const T *>) {}
  void bar4(fptr1<char * __restrict>);
  void t4() { foo4(&bar4); }
  // expected-error@-1 {{no matching function for call to 'foo4'}}
  // expected-note@-4  {{candidate template ignored: cannot deduce a type for 'T' that would make 'const T' equal 'char'}}

  template<typename T> void foo5(T(T)) {}
  const int bar5(int);
  void t5() { foo5(bar5); }
  // expected-error@-1 {{no matching function for call to 'foo5'}}
  // expected-note@-4  {{candidate template ignored: deduced conflicting types for parameter 'T' ('const int' vs. 'int')}}

  struct Foo6 {};
  template<typename T> void foo6(void(*)(struct Foo6, T)) {}
  void bar6(Foo6, int);
  void t6() { foo6(bar6); }
}
#else

// expected-no-diagnostics
namespace undefined_warnings {
  // Make sure we don't get an "undefined but used internal symbol" warning for the deduction guide here.
  namespace {
    template <typename T>
    struct TemplDObj {
      explicit TemplDObj(T func) noexcept {}
    };
    auto test1 = TemplDObj(0);

    TemplDObj(float) -> TemplDObj<double>;
    auto test2 = TemplDObj(.0f);
  }
}

namespace GH51710 {
template<typename T>
struct A {
  A(T f()) {}
  A(int f(), T) {}

  A(T array[10]) {}
  A(int array[10], T) {}
};

template<typename T>
struct B {
   B(T array[]) {}
   B(int array[], T) {}
};


int foo();

void bar() {
  A test1(foo);
  A test2(foo, 1);

  int array[10];
  A test3(array);
  A test4(array, 1);

  B test5(array);
  B test6(array, 1);
}
} // namespace GH51710

#endif
