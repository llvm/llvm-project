// RUN: %clang_cc1 -fsyntax-only -verify -std=c++1z %s

template<typename T, T val> struct A {};

template<typename T, typename U> constexpr bool is_same = false; // expected-note +{{here}}
template<typename T> constexpr bool is_same<T, T> = true;

namespace String {
  A<const char*, "test"> a; // expected-error {{does not refer to any declaration}}
  A<const char (&)[5], "test"> b; // expected-error {{does not refer to any declaration}}
}

namespace Array {
  char arr[3];
  char x;
  A<const char*, arr> a;
  A<const char(&)[3], arr> b;
  A<const char*, &arr[0]> c;
  A<const char*, &arr[1]> d; // expected-error {{refers to subobject '&arr[1]'}}
  A<const char*, (&arr)[0]> e;
  A<const char*, &x> f;
  A<const char*, &(&x)[0]> g;
  A<const char*, &(&x)[1]> h; // expected-error {{refers to subobject '&x + 1'}}
  A<const char*, 0> i; // expected-error {{not allowed in a converted constant}}
  A<const char*, nullptr> j;

  extern char aub[];
  A<char[], aub> k;
}

namespace Function {
  void f();
  void g() noexcept;
  void h();
  void h(int);
  template<typename...T> void i(T...);
  typedef A<void (*)(), f> a;
  typedef A<void (*)(), &f> a;
  typedef A<void (*)(), g> b;
  typedef A<void (*)(), &g> b;
  typedef A<void (*)(), h> c;
  typedef A<void (*)(), &h> c;
  typedef A<void (*)(), i> d;
  typedef A<void (*)(), &i> d;
  typedef A<void (*)(), i<>> d;
  typedef A<void (*)(), i<int>> e; // expected-error {{is not implicitly convertible}}

  typedef A<void (*)(), 0> x; // expected-error {{not allowed in a converted constant}}
  typedef A<void (*)(), nullptr> y;
}

void Func() {
  A<const char*, __func__> a; // expected-error {{does not refer to any declaration}}
}

namespace LabelAddrDiff {
  void f() {
    a: b: A<int, __builtin_constant_p(true) ? (__INTPTR_TYPE__)&&b - (__INTPTR_TYPE__)&&a : 0> s; // expected-error {{label address difference}}
  };
}

namespace Temp {
  struct S { int n; };
  constexpr S &addr(S &&s) { return s; }
  A<S &, addr({})> a; // expected-error {{constant}} expected-note 2{{temporary}}
  A<S *, &addr({})> b; // expected-error {{constant}} expected-note 2{{temporary}}
  A<int &, addr({}).n> c; // expected-error {{constant}} expected-note 2{{temporary}}
  A<int *, &addr({}).n> d; // expected-error {{constant}} expected-note 2{{temporary}}
}

namespace std { struct type_info; }

namespace RTTI {
  A<const std::type_info&, typeid(int)> a; // expected-error {{does not refer to any declaration}}
  A<const std::type_info*, &typeid(int)> b; // expected-error {{does not refer to any declaration}}
}

namespace PtrMem {
  struct B { int b; };
  struct C : B {};
  struct D : B {};
  struct E : C, D { int e; };

  constexpr int B::*b = &B::b;
  constexpr int C::*cb = b;
  constexpr int D::*db = b;
  constexpr int E::*ecb = cb; // expected-note +{{here}}
  constexpr int E::*edb = db; // expected-note +{{here}}

  constexpr int E::*e = &E::e;
  constexpr int D::*de = (int D::*)e;
  constexpr int C::*ce = (int C::*)e;
  constexpr int B::*bde = (int B::*)de; // expected-note +{{here}}
  constexpr int B::*bce = (int B::*)ce; // expected-note +{{here}}

  // FIXME: This should all be accepted, but we don't yet have a representation
  // nor mangling for this form of template argument.
  using Ab = A<int B::*, b>;
  using Ab = A<int B::*, &B::b>;
  using Abce = A<int B::*, bce>; // expected-error {{not supported}}
  using Abde = A<int B::*, bde>; // expected-error {{not supported}}
  static_assert(!is_same<Ab, Abce>, ""); // expected-error {{undeclared}} expected-error {{must be a type}}
  static_assert(!is_same<Ab, Abde>, ""); // expected-error {{undeclared}} expected-error {{must be a type}}
  static_assert(!is_same<Abce, Abde>, ""); // expected-error 2{{undeclared}} expected-error {{must be a type}}
  static_assert(is_same<Abce, A<int B::*, (int B::*)(int C::*)&E::e>, ""); // expected-error {{undeclared}} expected-error {{not supported}}

  using Ae = A<int E::*, e>;
  using Ae = A<int E::*, &E::e>;
  using Aecb = A<int E::*, ecb>; // expected-error {{not supported}}
  using Aedb = A<int E::*, edb>; // expected-error {{not supported}}
  static_assert(!is_same<Ae, Aecb>, ""); // expected-error {{undeclared}} expected-error {{must be a type}}
  static_assert(!is_same<Ae, Aedb>, ""); // expected-error {{undeclared}} expected-error {{must be a type}}
  static_assert(!is_same<Aecb, Aedb>, ""); // expected-error 2{{undeclared}} expected-error {{must be a type}}
  static_assert(is_same<Aecb, A<int E::*, (int E::*)(int C::*)&B::b>, ""); // expected-error {{undeclared}} expected-error {{not supported}}

  using An = A<int E::*, nullptr>;
  using A0 = A<int E::*, (int E::*)0>;
  static_assert(is_same<An, A0>);
}

namespace DeduceDifferentType {
  template<int N> struct A {};
  template<long N> int a(A<N>); // expected-note {{does not have the same type}}
  int a_imp = a(A<3>()); // expected-error {{no matching function}}
  int a_exp = a<3>(A<3>());

  template<decltype(nullptr)> struct B {};
  template<int *P> int b(B<P>); // expected-error {{value of type 'int *' is not implicitly convertible to 'decltype(nullptr)'}}
  int b_imp = b(B<nullptr>()); // expected-error {{no matching function}}
  int b_exp = b<nullptr>(B<nullptr>()); // expected-error {{no matching function}}

  struct X { constexpr operator int() { return 0; } } x;
  template<X &> struct C {};
  template<int N> int c(C<N>); // expected-error {{value of type 'int' is not implicitly convertible to 'DeduceDifferentType::X &'}}
  int c_imp = c(C<x>()); // expected-error {{no matching function}}
  int c_exp = c<x>(C<x>()); // expected-error {{no matching function}}

  struct Z;
  struct Y { constexpr operator Z&(); } y;
  struct Z { constexpr operator Y&() { return y; } } z;
  constexpr Y::operator Z&() { return z; }
  template<Y &> struct D {};
  template<Z &z> int d(D<z>); // expected-note {{couldn't infer template argument 'z'}}
  int d_imp = d(D<y>()); // expected-error {{no matching function}}
  int d_exp = d<y>(D<y>());
}

namespace DeclMatch {
  template<typename T, T> int f();
  template<typename T> class X { friend int f<T, 0>(); static int n; };
  template<typename T, T> int f() { return X<T>::n; }
  int k = f<int, 0>(); // ok, friend
}

namespace PR24921 {
  enum E { e };
  template<E> void f();
  template<int> void f(int);
  template<> void f<e>() {}
}

namespace Auto {
  namespace Basic {
    // simple auto
    template<auto x> constexpr auto constant = x; // expected-note {{declared here}}

    auto v1 = constant<5>;
    auto v2 = constant<true>;
    auto v3 = constant<'a'>;
    auto v4 = constant<2.5>;  // expected-error {{cannot have type 'double'}}

    using T1 = decltype(v1);
    using T1 = int;
    using T2 = decltype(v2);
    using T2 = bool;
    using T3 = decltype(v3);
    using T3 = char;

    // pointers
    template<auto v>    class B { };
    template<auto* p>   class B<p> { }; // expected-note {{matches}}
    template<auto** pp> class B<pp> { };
    template<auto* p0>   int &f(B<p0> b); // expected-note {{candidate}}
    template<auto** pp0> float &f(B<pp0> b); // expected-note {{candidate}}

    int a, *b = &a;
    int &r = f(B<&a>());
    float &s = f(B<&b>());

    // pointers to members
    template<typename T, auto *T::*p> struct B<p> {};
    template<typename T, auto **T::*p> struct B<p> {};
    template<typename T, auto *T::*p0>   char &f(B<p0> b); // expected-note {{candidate}}
    template<typename T, auto **T::*pp0> short &f(B<pp0> b); // expected-note {{candidate}}

    struct X { int n; int *p; int **pp; typedef int a, b; };
    auto t = f(B<&X::n>()); // expected-error {{no match}}
    char &u = f(B<&X::p>());
    short &v = f(B<&X::pp>());

    // A case where we need to do auto-deduction, and check whether the
    // resulting dependent types match during partial ordering. These
    // templates are not ordered due to the mismatching function parameter.
    template<typename T, auto *(*f)(T, typename T::a)> struct B<f> {}; // expected-note {{matches}}
    template<typename T, auto **(*f)(T, typename T::b)> struct B<f> {}; // expected-note {{matches}}
    int **g(X, int);
    B<&g> bg; // expected-error {{ambiguous}}
  }

  namespace Chained {
    // chained template argument deduction
    template<long n> struct C { };
    template<class T> struct D;
    template<class T, T n> struct D<C<n>>
    {
        using Q = T;
    };
    using DQ = long;
    using DQ = D<C<short(2)>>::Q;

    // chained template argument deduction from an array bound
    template<typename T> struct E;
    template<typename T, T n> struct E<int[n]> {
        using Q = T;
    };
    using EQ = E<int[short(42)]>::Q;
    using EQ = decltype(sizeof 0);

    template<int N> struct F;
    template<typename T, T N> int foo(F<N> *) = delete;  // expected-note {{explicitly deleted}}
    void foo(void *); // expected-note {{candidate function}}
    void bar(F<0> *p) {
        foo(p); // expected-error {{deleted function}}
    }
  }

  namespace ArrayToPointer {
    constexpr char s[] = "test";
    template<const auto* p> struct S { };
    S<s> p;

    template<typename R, typename P, R F(P)> struct A {};
    template<typename R, typename P, R F(P)> void x(A<R, P, F> a);
    void g(int) { x(A<void, int, &g>()); }
  }

  namespace DecltypeAuto {
    template<auto v> struct A { };
    template<decltype(auto) v> struct DA { };
    template<auto&> struct R { };

    auto n = 0; // expected-note + {{declared here}}
    A<n> a; // expected-error {{not a constant}} expected-note {{non-const variable 'n'}}
    DA<n> da1;  // expected-error {{not a constant}} expected-note {{non-const variable 'n'}}
    DA<(n)> da2;
    R<n> r;
  }

  namespace Decomposition {
    // Types of deduced non-type template arguments must match exactly, so
    // partial ordering fails in both directions here.
    template<auto> struct Any;
    template<int N> struct Any<N> { typedef int Int; }; // expected-note 3{{match}}
    template<short N> struct Any<N> { typedef int Short; }; // expected-note 3{{match}}
    Any<0>::Int is_int; // expected-error {{ambiguous}}
    Any<(short)0>::Short is_short; // expected-error {{ambiguous}}
    Any<(char)0>::Short is_char; // expected-error {{ambiguous}}

    template<int, auto> struct NestedAny;
    template<auto N> struct NestedAny<0, N>; // expected-note 3{{match}}
    template<int N> struct NestedAny<0, N> { typedef int Int; }; // expected-note 3{{match}}
    template<short N> struct NestedAny<0, N> { typedef int Short; }; // expected-note 3{{match}}
    NestedAny<0, 0>::Int nested_int; // expected-error {{ambiguous}}
    NestedAny<0, (short)0>::Short nested_short; // expected-error {{ambiguous}}
    NestedAny<0, (char)0>::Short nested_char; // expected-error {{ambiguous}}

    double foo(int, bool);
    template<auto& f> struct fn_result_type;

    template<class R, class... Args, R (& f)(Args...)>
    struct fn_result_type<f>
    {
        using type = R;
    };

    using R1 = fn_result_type<foo>::type;
    using R1 = double;

    template<int, auto &f> struct fn_result_type_partial_order;
    template<auto &f> struct fn_result_type_partial_order<0, f>;
    template<class R, class... Args, R (& f)(Args...)>
    struct fn_result_type_partial_order<0, f> {};
    fn_result_type_partial_order<0, foo> frtpo;
  }

  namespace Variadic {
    template<auto... vs> struct value_list { };

    using size_t = decltype(sizeof 0);
    template<size_t n, class List> struct nth_element;
    template<size_t n, class List> constexpr auto nth_element_v = nth_element<n, List>::value;

    template<size_t n, auto v0, auto... vs>
    struct nth_element<n, value_list<v0, vs...>>
    {
        static constexpr auto value = nth_element<n - 1, value_list<vs...>>::value;
    };
    template<auto v0, auto... vs>
    struct nth_element<0, value_list<v0, vs...>>
    {
        static constexpr auto value = v0;
    };

    static_assert(nth_element_v<2, value_list<'a', 27U, false>> == false, "value mismatch");
  }
}

namespace Nested {
  template<typename T> struct A {
    template<auto X> struct B;
    template<auto *P> struct B<P>;
    template<auto **P> struct B<P> { using pointee = decltype(+**P); };
    template<auto (*P)(T)> struct B<P> { using param = T; };
    template<typename U, auto (*P)(T, U)> struct B<P> { using param2 = U; };
  };

  using Int = int;

  int *n;
  using Int = A<int>::B<&n>::pointee;

  void f(int);
  using Int = A<int>::B<&f>::param;

  void g(int, int);
  using Int = A<int>::B<&g>::param2;
}
