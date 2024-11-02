// RUN: %clang_cc1 -fsyntax-only -verify -std=c++1z %s

template<typename T, T val> struct A {};

template<typename T, typename U> constexpr bool is_same = false; // expected-note +{{here}}
template<typename T> constexpr bool is_same<T, T> = true;

namespace String {
  A<const char*, "test"> a; // expected-error {{pointer to subobject of string literal}}
  A<const char (&)[5], "test"> b; // expected-error {{reference to string literal}}
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
  typedef A<void (*)(), i<int>> e; // expected-error {{value of type '<overloaded function type>' is not implicitly convertible to 'void (*)()'}}

  typedef A<void (*)(), 0> x; // expected-error {{not allowed in a converted constant}}
  typedef A<void (*)(), nullptr> y;
}

void Func() {
  A<const char*, __func__> a; // expected-error {{pointer to subobject of predefined '__func__' variable}}
}

namespace LabelAddrDiff {
  void f() {
    a: b: A<int, __builtin_constant_p(true) ? (__INTPTR_TYPE__)&&b - (__INTPTR_TYPE__)&&a : 0> s; // expected-error {{label address difference}}
  };
}

namespace Temp {
  struct S { int n; };
  constexpr S &addr(S &&s) { return s; }
  A<S &, addr({})> a; // expected-error {{reference to temporary object}}
  A<S *, &addr({})> b; // expected-error {{pointer to temporary object}}
  A<int &, addr({}).n> c; // expected-error {{reference to subobject of temporary object}}
  A<int *, &addr({}).n> d; // expected-error {{pointer to subobject of temporary object}}
}

namespace std { struct type_info; }

namespace RTTI {
  A<const std::type_info&, typeid(int)> a; // expected-error {{reference to type_info object}}
  A<const std::type_info*, &typeid(int)> b; // expected-error {{pointer to type_info object}}
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
  static_assert(is_same<Abce, A<int B::*, (int B::*)(int C::*)&E::e>>, ""); // expected-error {{undeclared}} expected-error {{not supported}}

  using Ae = A<int E::*, e>;
  using Ae = A<int E::*, &E::e>;
  using Aecb = A<int E::*, ecb>; // expected-error {{not supported}}
  using Aedb = A<int E::*, edb>; // expected-error {{not supported}}
  static_assert(!is_same<Ae, Aecb>, ""); // expected-error {{undeclared}} expected-error {{must be a type}}
  static_assert(!is_same<Ae, Aedb>, ""); // expected-error {{undeclared}} expected-error {{must be a type}}
  static_assert(!is_same<Aecb, Aedb>, ""); // expected-error 2{{undeclared}} expected-error {{must be a type}}
  static_assert(is_same<Aecb, A<int E::*, (int E::*)(int C::*)&B::b>>, ""); // expected-error {{undeclared}} expected-error {{not supported}}

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
  template<int N> int c(C<N>); // expected-error {{value of type 'int' is not implicitly convertible to 'X &'}}
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

    void type_affects_identity(B<&a>) {}
    void type_affects_identity(B<(const int*)&a>) {}
    void type_affects_identity(B<(void*)&a>) {}
    void type_affects_identity(B<(const void*)&a>) {}

    // pointers to members
    template<typename T, auto *T::*p> struct B<p> {};
    template<typename T, auto **T::*p> struct B<p> {};
    template<typename T, auto *T::*p0>   char &f(B<p0> b); // expected-note {{candidate}}
    template<typename T, auto **T::*pp0> short &f(B<pp0> b); // expected-note {{candidate}}

    struct X { int n; int *p; int **pp; typedef int a, b; };
    auto t = f(B<&X::n>()); // expected-error {{no match}}
    char &u = f(B<&X::p>());
    short &v = f(B<&X::pp>());

    struct Y : X {};
    void type_affects_identity(B<&X::n>) {}
    void type_affects_identity(B<(int Y::*)&X::n>) {} // FIXME: expected-error {{sorry}}
    void type_affects_identity(B<(const int X::*)&X::n>) {}
    void type_affects_identity(B<(const int Y::*)&X::n>) {} // FIXME: expected-error {{sorry}}

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

namespace rdar41852459 {
template <auto V> struct G {};

template <class T> struct S {
  template <auto V> void f() {
    G<V> x;
  }
  template <auto *PV> void f2() {
    G<PV> x;
  }
  template <decltype(auto) V> void f3() {
    G<V> x;
  }
};

template <auto *PV> struct I {};

template <class T> struct K {
  template <auto *PV> void f() {
    I<PV> x;
  }
  template <auto V> void f2() {
    I<V> x;
  }
  template <decltype(auto) V> void f3() {
    I<V> x;
  }
};

template <decltype(auto)> struct L {};
template <class T> struct M {
  template <auto *PV> void f() {
    L<PV> x;
  }
  template <auto V> void f() {
    L<V> x;
  }
  template <decltype(auto) V> void f() {
    L<V> x;
  }
};
}

namespace PR42362 {
  template<auto ...A> struct X { struct Y; void f(int...[A]); };
  template<auto ...A> struct X<A...>::Y {};
  template<auto ...A> void X<A...>::f(int...[A]) {}
  void f() { X<1, 2>::Y y; X<1, 2>().f(0, 0); }

  template<typename, auto...> struct Y;
  template<auto ...A> struct Y<int, A...> {};
  Y<int, 1, 2, 3> y;

  template<auto (&...F)()> struct Z { struct Q; };
  template<auto (&...F)()> struct Z<F...>::Q {};
  Z<f, f, f>::Q q;
}

namespace QualConv {
  int *X;
  template<const int *const *P> void f() {
    using T = decltype(P);
    using T = const int* const*;
  }
  template void f<&X>();

  template<const int *const &R> void g() {
    using T = decltype(R);
    using T = const int *const &;
  }
  template void g<(const int *const&)X>();
}

namespace FunctionConversion {
  struct a { void c(char *) noexcept; };
  template<void (a::*f)(char*)> void g() {
    using T = decltype(f);
    using T = void (a::*)(char*); // (not 'noexcept')
  }
  template void g<&a::c>();

  void c() noexcept;
  template<void (*p)()> void h() {
    using T = decltype(p);
    using T = void (*)(); // (not 'noexcept')
  }
  template void h<&c>();
}

namespace VoidPtr {
  // Note, this is an extension in C++17 but valid in C++20.
  template<void *P> void f() {
    using T = decltype(P);
    using T = void*;
  }
  int n;
  template void f<(void*)&n>();
}

namespace PR42108 {
  struct R {};
  struct S { constexpr S() {} constexpr S(R) {} };
  struct T { constexpr operator S() { return {}; } };
  template <const S &> struct A {};
  void f() {
    A<R{}>(); // expected-error {{would bind reference to a temporary}}
    A<S{}>(); // expected-error {{reference to temporary object}}
    A<T{}>(); // expected-error {{reference to temporary object}}
  }
}

namespace PR46637 {
  template<auto (*f)() -> auto> struct X { // expected-note {{here}}
    auto call() { return f(); }
  };
  X<nullptr> x; // expected-error {{incompatible initializer}}

  void *f();
  X<f> y;
  int n = y.call(); // expected-error {{cannot initialize a variable of type 'int' with an rvalue of type 'void *'}}
}

namespace PR48517 {
  template<const int *P> struct A { static constexpr const int *p = P; };
  template<typename T> auto make_nonconst() {
    static int n;
    return A<&n>();
  };
  using T = decltype(make_nonconst<int>()); // expected-note {{previous}}
  using U = decltype(make_nonconst<float>());
  static_assert(T::p != U::p);
  using T = U; // expected-error {{different types}}

  template<typename T> auto make_const() {
    static constexpr int n = 42;
    return A<&n>();
  };
  using V = decltype(make_const<int>()); // expected-note {{previous}}
  using W = decltype(make_const<float>());
  static_assert(*V::p == *W::p);
  static_assert(V::p != W::p);
  using V = W; // expected-error {{different types}}

  template<auto V> struct Q {
    using X = int;
    static_assert(V == "primary template should not be instantiated");
  };
  template<typename T> struct R {
    int n;
    constexpr int f() {
      return Q<&R::n>::X;
    }
  };
  template<> struct Q<&R<int>::n> { static constexpr int X = 1; };
  static_assert(R<int>().f() == 1);
}

namespace dependent_reference {
  template<int &r> struct S { int *q = &r; };
  template<int> auto f() { static int n; return S<n>(); }
  auto v = f<0>();
  auto w = f<1>();
  static_assert(!is_same<decltype(v), decltype(w)>);
  // Ensure that we can instantiate the definition of S<...>.
  int n = *v.q + *w.q;
}

namespace decay {
  template<typename T, typename C, const char *const A[(int)T::count]> struct X {
    template<typename CC> void f(const X<T, CC, A> &v) {}
  };
  struct A {
    static constexpr const char *arr[] = {"hello", "world"};
    static constexpr int count = 2;
  };
  void f() {
    X<A, int, A::arr> x1;
    X<A, float, A::arr> x2;
    x1.f(x2);
  }
}

namespace TypeSuffix {
  template <auto N> struct A {};
  template <> struct A<1> { using type = int; }; // expected-note {{'A<1>::type' declared here}}
  A<1L>::type a;                                 // expected-error {{no type named 'type' in 'TypeSuffix::A<1L>'; did you mean 'A<1>::type'?}}

  template <auto N> struct B {};
  template <> struct B<1> { using type = int; }; // expected-note {{'B<1>::type' declared here}}
  B<2>::type b;                                  // expected-error {{no type named 'type' in 'TypeSuffix::B<2>'; did you mean 'B<1>::type'?}}

  template <auto N> struct C {};
  template <> struct C<'a'> { using type = signed char; }; // expected-note {{'C<'a'>::type' declared here}}
  C<(signed char)'a'>::type c;                             // expected-error {{no type named 'type' in 'TypeSuffix::C<(signed char)'a'>'; did you mean 'C<'a'>::type'?}}

  template <auto N> struct D {};
  template <> struct D<'a'> { using type = signed char; }; // expected-note {{'D<'a'>::type' declared here}}
  D<'b'>::type d;                                          // expected-error {{no type named 'type' in 'TypeSuffix::D<'b'>'; did you mean 'D<'a'>::type'?}}

  template <auto N> struct E {};
  template <> struct E<'a'> { using type = unsigned char; }; // expected-note {{'E<'a'>::type' declared here}}
  E<(unsigned char)'a'>::type e;                             // expected-error {{no type named 'type' in 'TypeSuffix::E<(unsigned char)'a'>'; did you mean 'E<'a'>::type'?}}

  template <auto N> struct F {};
  template <> struct F<'a'> { using type = unsigned char; }; // expected-note {{'F<'a'>::type' declared here}}
  F<'b'>::type f;                                            // expected-error {{no type named 'type' in 'TypeSuffix::F<'b'>'; did you mean 'F<'a'>::type'?}}

  template <auto... N> struct X {};
  X<1, 1u>::type y; // expected-error {{no type named 'type' in 'TypeSuffix::X<1, 1U>'}}
  X<1, 1>::type z; // expected-error {{no type named 'type' in 'TypeSuffix::X<1, 1>'}}
}

namespace no_crash {
template <class T>
class Base {
public:
  template <class> class EntryPointSpec {};
  template <auto Method>
  using EntryPoint = EntryPointSpec<T>;
};

class Derived : Base<Derived>{
  template <class...> class Spec {};

  void Invalid(Undefined) const; // expected-error {{unknown type name 'Undefined'}}
  void crash() {
    return Spec{
        EntryPoint<&Invalid>()
    };
  }
};
} // no_crash

namespace PR47792 {
  using I = int;

  template<decltype(auto)> int a;
  const int n = 0;
  const I n2 = 0;
  static_assert(&a<n> == &a<0>, "both should have type 'int'");
  static_assert(&a<n2> == &a<0>, "both should have type 'int'");

  int m;
  const int &r1 = m;
  int &r2 = m;
  static_assert(&a<r1> != &a<r2>, "should have different types");

  const I &r3 = m;
  static_assert(&a<r1> == &a<r3>, "should have different types");
  static_assert(&a<r2> != &a<r3>, "should have different types");

  void foo();
  template <void () = foo> void bar() {}
  template void bar<>();    // expected-note {{previous explicit instantiation is here}}
  template void bar<foo>(); // expected-error {{duplicate explicit instantiation of 'bar<&PR47792::foo>'}}
}
