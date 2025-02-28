// RUN: %clang_cc1 -fsyntax-only -triple x86_64-unknown-linux -Wno-c++11-narrowing -Wno-literal-conversion -std=c++20 -verify %s

namespace test1 {
template <typename T>
struct Foo { T t; };
template <typename U>
using Bar = Foo<U>;

Bar s = {1};
}  // namespace test1

namespace test2 {
template <typename X, typename Y>
struct XYpair {
  X x;
  Y y;
};
// A tricky explicit deduction guide that swapping X and Y.
template <typename X, typename Y>
XYpair(X, Y) -> XYpair<Y, X>;
template <typename U, typename V>
using AliasXYpair = XYpair<U, V>;

AliasXYpair xy = {1.1, 2};  // XYpair<int, double>
static_assert(__is_same(decltype(xy.x), int));
static_assert(__is_same(decltype(xy.y), double));
}  // namespace test2

namespace test3 {
template <typename T, class>
struct container {
  // test with default arguments.
  container(T a, T b = T());
};

template <class T>
using vector = container<T, int>;
vector v(0, 0);
}  // namespace test3

namespace test4 {
// Explicit deduction guide.
template <class T>
struct X {
  T t;
  X(T);
};

template <class T>
X(T) -> X<double>;

template <class T>
using AX = X<T>;

AX s = {1};
static_assert(__is_same(decltype(s.t), double)); // explicit one is picked.
}  // namespace test4

namespace test5 {
template <int B>
struct Foo {};
// Template parameter pack
template <int... C>
using AF = Foo<1>;
auto a = AF{};
}  // namespace test5

namespace test6 {
// non-type template argument.
template <typename T, bool B = false>
struct Foo {
  Foo(T);
};
template <typename T>
using AF = Foo<T, 1>;

AF b{0};
}  // namespace test6

namespace test7 {
template <typename T>
struct Foo {
  Foo(T);
};
// using alias chain.
template <typename U>
using AF1 = Foo<U>;
template <typename K>
using AF2 = AF1<K>;
AF2 b = 1;
}  // namespace test7

namespace test8 {
template <typename T, int N>
struct Foo {
  Foo(T const (&)[N]);
};

template <typename X, int Y>
using Bar = Foo<X, Y>;

Bar s = {{1}};
}  // namespace test8

namespace test9 {
template <typename T, int N>
struct Foo {
  Foo(T const (&)[N]);
};

template <typename X, int Y>
using Bar = Foo<X, sizeof(X)>; // expected-note {{candidate template ignored: couldn't infer template argument 'X'}} \
                               // expected-note {{implicit deduction guide declared as 'template <typename X> requires __is_deducible(test9::Bar, Foo<X, sizeof(X)>) Bar(Foo<X, sizeof(X)>) -> Foo<X, sizeof(X)>'}} \
                               // expected-note {{implicit deduction guide declared as 'template <typename X> requires __is_deducible(test9::Bar, Foo<X, sizeof(X)>) Bar(const X (&)[sizeof(X)]) -> Foo<X, sizeof(X)>'}} \
                               // expected-note {{candidate template ignored: constraints not satisfied [with X = int]}} \
                               // expected-note {{cannot deduce template arguments for 'Bar' from 'Foo<int, 4UL>'}}


Bar s = {{1}}; // expected-error {{no viable constructor or deduction guide }}
}  // namespace test9

namespace test10 {
template <typename T>
struct Foo {
  template <typename U>
  Foo(U);
};

template <typename U>
Foo(U) -> Foo<U*>;

template <typename K>
using A = Foo<K>;
A a(2);  // Foo<int*>
}  // namespace test10

namespace test11 {
struct A {};
template<class T> struct Foo { T c; };
template<class X, class Y=A>
using AFoo = Foo<Y>; // expected-note {{candidate template ignored: could not match 'Foo<Y>' against 'int'}} \
                    // expected-note {{implicit deduction guide declared as 'template <class Y = A> requires __is_deducible(test11::AFoo, Foo<Y>) AFoo(Foo<Y>) -> Foo<Y>'}} \
                    // expected-note {{candidate template ignored: constraints not satisfied [with Y = int]}} \
                    // expected-note {{cannot deduce template arguments for 'AFoo' from 'Foo<int>'}} \
                    // expected-note {{implicit deduction guide declared as 'template <class Y = A> requires __is_deducible(test11::AFoo, Foo<Y>) AFoo(Y) -> Foo<Y>'}} \
                    // expected-note {{candidate function template not viable: requires 0 arguments, but 1 was provided}} \
                    // expected-note {{implicit deduction guide declared as 'template <class Y = A> requires __is_deducible(test11::AFoo, Foo<Y>) AFoo() -> Foo<Y>'}}

AFoo s = {1}; // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'AFoo'}}
} // namespace test11

namespace test12 {
// no crash on null access attribute
template<typename X>
struct Foo {
  template<typename K>
  struct Bar {
    Bar(K);
  };

  template<typename U>
  using ABar = Bar<U>;
  void test() { ABar k = 2; }
};

void func(Foo<int> s) {
  s.test();
}
} // namespace test12

namespace test13 {
template <typename... Ts>
struct Foo {
  Foo(Ts...);
};

template <typename... Ts>
using AFoo = Foo<Ts...>;

auto b = AFoo{};
AFoo a(1, 2);

template <typename T>
using BFoo = Foo<T, T>;
BFoo b2(1.0, 2.0);
} // namespace test13

namespace test14 {
template<typename T>
concept IsInt = __is_same(decltype(T()), int);

template<IsInt T, int N>
struct Foo {
  Foo(T const (&)[N]);
};

template <int K>
using Bar = Foo<double, K>; // expected-note {{constraints not satisfied for class template 'Foo'}}
// expected-note@-1 {{candidate template ignored: could not match}}
// expected-note@-2 {{implicit deduction guide declared as 'template <int K> requires __is_deducible(test14::Bar, Foo<double, K>) Bar(Foo<double, K>) -> Foo<double, K>'}}
// expected-note@-3 {{implicit deduction guide declared as 'template <int K> requires __is_deducible(test14::Bar, Foo<double, K>) Bar(const double (&)[K]) -> Foo<double, K>'}}
double abc[3];
Bar s2 = {abc}; // expected-error {{no viable constructor or deduction guide for deduction }}
} // namespace test14

namespace test15 {
template <class T> struct Foo { Foo(T); };

template<class V> using AFoo = Foo<V *>;
template<typename> concept False = false;
template<False W>
using BFoo = AFoo<W>; // expected-note {{candidate template ignored: constraints not satisfied [with V = int]}} \
                      // expected-note {{cannot deduce template arguments for 'BFoo' from 'Foo<int *>'}} \
                      // expected-note {{implicit deduction guide declared as 'template <class V> requires __is_deducible(AFoo, Foo<V *>) && __is_deducible(test15::BFoo, Foo<V *>) BFoo(V *) -> Foo<V *>}} \
                      // expected-note {{candidate template ignored: could not match 'Foo<V *>' against 'int *'}} \
                      // expected-note {{template <class V> requires __is_deducible(AFoo, Foo<V *>) && __is_deducible(test15::BFoo, Foo<V *>) BFoo(Foo<V *>) -> Foo<V *>}}
int i = 0;
AFoo a1(&i); // OK, deduce Foo<int *>

// the W is not deduced from the deduced type Foo<int *>.
BFoo b2(&i); // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'BFoo'}}
} // namespace test15

namespace test16 {
struct X { X(int); X(const X&); };
template<class T>
struct Foo {
  T t;
  Foo(T t) : t(t) {}
};
template<class T>
using AFoo = Foo<T>;
int i = 0;
AFoo s{i};
static_assert(__is_same(decltype(s.t), int));

template<class T>
using BFoo = AFoo<T>;

// template explicit deduction guide.
template<class T>
Foo(T) -> Foo<float>;
static_assert(__is_same(decltype(AFoo(i).t), float));
static_assert(__is_same(decltype(BFoo(i).t), float));

// explicit deduction guide.
Foo(int) -> Foo<X>;
static_assert(__is_same(decltype(AFoo(i).t), X));
static_assert(__is_same(decltype(BFoo(i).t), X));

Foo(double) -> Foo<int>;
static_assert(__is_same(decltype(AFoo(1.0).t), int));
static_assert(__is_same(decltype(BFoo(1.0).t), int));
} // namespace test16

namespace test17 {
template <typename T>
struct Foo { T t; };

// CTAD for alias templates only works for the RHS of the alias of form of
//  [typename] [nested-name-specifier] [template] simple-template-id
template <typename U>
using AFoo = Foo<U>*; // expected-note {{template is declared here}}

AFoo s = {1}; // expected-error {{alias template 'AFoo' requires template arguments; argument deduction only allowed for}}
} // namespace test17

namespace test18 {
template<typename T>
concept False = false; // expected-note {{because 'false' evaluated to false}}

template <typename T> struct Foo { T t; };

template<typename T> requires False<T> // expected-note {{because 'int' does not satisfy 'False'}}
Foo(T) -> Foo<int>;

template <typename U>
using Bar = Foo<U>; // expected-note {{could not match 'Foo<U>' against 'int'}} \
                    // expected-note {{implicit deduction guide declared as 'template <typename U> requires __is_deducible(test18::Bar, Foo<U>) Bar(Foo<U>) -> Foo<U>'}} \
                    // expected-note {{candidate template ignored: constraints not satisfied}} \
                    // expected-note {{implicit deduction guide declared as 'template <typename T> requires False<T> && __is_deducible(test18::Bar, Foo<int>) Bar(T) -> Foo<int>'}} \
                    // expected-note {{candidate function template not viable}} \
                    // expected-note {{implicit deduction guide declared as 'template <typename U> requires __is_deducible(test18::Bar, Foo<U>) Bar() -> Foo<U>'}}

Bar s = {1}; // expected-error {{no viable constructor or deduction guide for deduction of template arguments}}
} // namespace test18

// GH85406, verify no crash on invalid alias templates.
namespace test19 {
template <typename T>
class Foo {};

template <typename T>
template <typename K>
using Bar2 = Foo<K>; // expected-error {{extraneous template parameter list in alias template declaration}}

Bar2 b = 1; // expected-error {{no viable constructor or deduction guide for deduction of template arguments}}
} // namespace test19

// GH85385
namespace test20 {
template <template <typename> typename T>
struct K {};

template <typename U>
class Foo {};

// Verify that template template type parameter TTP is referenced/used in the
// template arguments of the RHS.
template <template<typename> typename TTP>
using Bar = Foo<K<TTP>>; // expected-note {{candidate template ignored: could not match 'Foo<K<TTP>>' against 'int'}} \
                        // expected-note {{implicit deduction guide declared as 'template <template <typename> typename TTP> requires __is_deducible(test20::Bar, Foo<K<TTP>>) Bar(Foo<K<TTP>>) -> Foo<K<TTP>>'}}

template <class T>
class Container {};
Bar t = Foo<K<Container>>();

Bar s = 1; // expected-error {{no viable constructor or deduction guide for deduction of template arguments of}}
} // namespace test20

namespace test21 {
template <typename T, unsigned N>
struct Array { const T member[N]; };
template <unsigned N>
using String = Array<char, N>;

// Verify no crash on constructing the aggregate deduction guides.
String s("hello");
} // namespace test21

// GH89013
namespace test22 {
class Base {};
template <typename T>
class Derived final : public Base {};

template <typename T, typename D>
requires __is_base_of(Base, D)
struct Foo {
  explicit Foo(D) {}
};

template <typename U>
using AFoo = Foo<int, Derived<U>>;

AFoo a(Derived<int>{});
} // namespace test22

namespace test23 {
// We have an aggregate deduction guide "G(T) -> G<T>".
template<typename T>
struct G { T t1; };

template<typename X = int>
using AG = G<int>;

AG ag(1.0);
// Verify that the aggregate deduction guide "AG(int) -> AG<int>" is built and
// choosen.
static_assert(__is_same(decltype(ag.t1), int));
} // namespace test23

// GH90177
// verify that the transformed require-clause of the alias deduction gudie has
// the right depth info.
namespace test24 {
class Forward;
class Key {};

template <typename D>
constexpr bool C = sizeof(D);

// Case1: the alias template and the underlying deduction guide are in the same
// scope.
template <typename T>
struct Case1 {
  template <typename U>
  struct Foo {
    Foo(U);
  };

  template <typename V>
  requires (C<V>)
  Foo(V) -> Foo<V>;

  template <typename Y>
  using Alias = Foo<Y>;
};
// The require-clause should be evaluated on the type Key.
Case1<Forward>::Alias t2 = Key();


// Case2: the alias template and underlying deduction guide are in different
// scope.
template <typename T>
struct Foo {
  Foo(T);
};
template <typename U>
requires (C<U>)
Foo(U) -> Foo<U>;

template <typename T>
struct Case2 {
  template <typename Y>
  using Alias = Foo<Y>;
};
// The require-caluse should be evaluated on the type Key.
Case2<Forward>::Alias t1 = Key();

// Case3: crashes on the constexpr evaluator due to the mixed-up depth in
// require-expr.
template <class T1>
struct A1 {
  template<class T2>
  struct A2 {
    template <class T3>
    struct Foo {
      Foo(T3);
    };
    template <class T3>
    requires C<T3>
    Foo(T3) -> Foo<T3>;
  };
};
template <typename U>
using AFoo = A1<int>::A2<int>::Foo<U>;
AFoo case3(1);

// Case4: crashes on the constexpr evaluator due to the mixed-up index for the
// template parameters `V`.
template<class T, typename T2>
struct Case4 {
  template<class V> requires C<V>
  Case4(V, T);
};

template<class T2>
using ACase4 = Case4<T2, T2>;
ACase4 case4{0, 1};

} // namespace test24

namespace GH92212 {
template<typename T, typename...Us>
struct A{
  template<typename V> requires __is_same(V, int)
  A(V);
};

template<typename...TS>
using AA = A<int, TS...>;
AA a{0};
}

namespace GH94927 {
template <typename T>
struct A {
  A(T);
};
A(int) -> A<char>;

template <typename U>
using B1 = A<U>;
B1 b1(100); // deduce to A<char>;
static_assert(__is_same(decltype(b1), A<char>));

template <typename U>
requires (!__is_same(U, char)) // filter out the explicit deduction guide.
using B2 = A<U>;
template <typename V>
using B3 = B2<V>;

B2 b2(100); // deduced to A<int>;
static_assert(__is_same(decltype(b2), A<int>));
B3 b3(100); // decuded to A<int>;
static_assert(__is_same(decltype(b3), A<int>));


// the nested case
template <typename T1>
struct Out {
  template <typename T2>
  struct A {
    A(T2);
  };
  A(int) -> A<T1>;
  
  template <typename T3>
  using B = A<T3>;
};

Out<float>::B out(100); // deduced to Out<float>::A<float>;
static_assert(__is_same(decltype(out), Out<float>::A<float>));
}

namespace GH111508 {

template <typename V> struct S {
  using T = V;
  T Data;
};

template <typename V> using Alias = S<V>;

Alias A(42);

} // namespace GH111508

namespace GH113518 {

template <class T, unsigned N> struct array {
  T value[N];
};

template <typename Tp, typename... Up>
array(Tp, Up...) -> array<Tp, 1 + sizeof...(Up)>;

template <typename T> struct ArrayType {
  template <unsigned size> using Array = array<T, size>;
};

template <ArrayType<int>::Array array> void test() {}

void foo() { test<{1, 2, 3}>(); }

} // namespace GH113518

namespace GH125821 {
template<typename T>
struct A { A(T){} };

template<typename T>
using Proxy = T;

template<typename T>
using C = Proxy< A<T> >;

C test{ 42 }; // expected-error {{no viable constructor or deduction guide for deduction of template arguments}}

} // namespace GH125821
