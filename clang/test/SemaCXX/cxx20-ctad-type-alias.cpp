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
                               // expected-note {{implicit deduction guide declared as 'template <typename X> requires __is_deducible(test9::Bar, test9::Foo<X, sizeof(X)>) Bar(test9::Foo<X, sizeof(X)>) -> test9::Foo<X, sizeof(X)>'}} \
                               // expected-note {{implicit deduction guide declared as 'template <typename X> requires __is_deducible(test9::Bar, test9::Foo<X, sizeof(X)>) Bar(const X (&)[sizeof(X)]) -> test9::Foo<X, sizeof(X)>'}} \
                               // expected-note {{candidate template ignored: constraints not satisfied [with X = int]}} \
                               // expected-note {{cannot deduce template arguments for 'test9::Bar' from 'test9::Foo<int, sizeof(int)>'}}


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
using AFoo = Foo<Y>; // expected-note {{candidate template ignored: could not match 'test11::Foo<Y>' against 'int'}} \
                    // expected-note {{implicit deduction guide declared as 'template <class Y = A> requires __is_deducible(test11::AFoo, test11::Foo<Y>) AFoo(test11::Foo<Y>) -> test11::Foo<Y>'}} \
                    // expected-note {{candidate template ignored: constraints not satisfied [with Y = int]}} \
                    // expected-note {{cannot deduce template arguments for 'test11::AFoo' from 'test11::Foo<int>'}} \
                    // expected-note {{implicit deduction guide declared as 'template <class Y = A> requires __is_deducible(test11::AFoo, test11::Foo<Y>) AFoo(Y) -> test11::Foo<Y>'}} \
                    // expected-note {{candidate function template not viable: requires 0 arguments, but 1 was provided}} \
                    // expected-note {{implicit deduction guide declared as 'template <class Y = A> requires __is_deducible(test11::AFoo, test11::Foo<Y>) AFoo() -> test11::Foo<Y>'}}

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
// expected-note@-1 {{candidate template ignored: could not match}} expected-note@-1 {{candidate template ignored: constraints not satisfied}}
// expected-note@-2 {{implicit deduction guide declared as 'template <int K> requires __is_deducible(test14::Bar, test14::Foo<double, K>) Bar(test14::Foo<double, K>) -> test14::Foo<double, K>'}}
// expected-note@-3 {{implicit deduction guide declared as 'template <int K> requires __is_deducible(test14::Bar, test14::Foo<double, K>) Bar(const double (&)[K]) -> test14::Foo<double, K>'}}
double abc[3];
Bar s2 = {abc}; // expected-error {{no viable constructor or deduction guide for deduction }}
} // namespace test14

namespace test15 {
template <class T> struct Foo { Foo(T); };

template<class V> using AFoo = Foo<V *>;
template<typename> concept False = false; // #test15_False
template<False W>
using BFoo = AFoo<W>; // expected-note {{candidate template ignored: constraints not satisfied [with W = int]}} \
                      // expected-note@-1 {{because 'int' does not satisfy 'False'}} \
                      // expected-note@#test15_False {{because 'false' evaluated to false}} \
                      // expected-note {{implicit deduction guide declared as 'template <False<> W> requires __is_deducible(test15::AFoo, test15::Foo<W *>) && __is_deducible(test15::BFoo, test15::Foo<W *>) BFoo(W *) -> test15::Foo<W *>}} \
                      // expected-note {{candidate template ignored: could not match 'test15::Foo<W *>' against 'int *'}} \
                      // expected-note {{template <False<> W> requires __is_deducible(test15::AFoo, test15::Foo<W *>) && __is_deducible(test15::BFoo, test15::Foo<W *>) BFoo(test15::Foo<W *>) -> test15::Foo<W *>}}
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
using Bar = Foo<U>; // expected-note {{could not match 'test18::Foo<U>' against 'int'}} \
                    // expected-note {{implicit deduction guide declared as 'template <typename U> requires __is_deducible(test18::Bar, test18::Foo<U>) Bar(test18::Foo<U>) -> test18::Foo<U>'}} \
                    // expected-note {{candidate template ignored: constraints not satisfied}} \
                    // expected-note {{implicit deduction guide declared as 'template <typename T> requires False<T> && __is_deducible(test18::Bar, Foo<int>) Bar(T) -> Foo<int>'}} \
                    // expected-note {{candidate function template not viable}} \
                    // expected-note {{implicit deduction guide declared as 'template <typename U> requires __is_deducible(test18::Bar, test18::Foo<U>) Bar() -> test18::Foo<U>'}}

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
using Bar = Foo<K<TTP>>; // expected-note {{candidate template ignored: could not match 'test20::Foo<K<TTP>>' against 'int'}} \
                        // expected-note {{implicit deduction guide declared as 'template <template <typename> typename TTP> requires __is_deducible(test20::Bar, test20::Foo<K<TTP>>) Bar(test20::Foo<K<TTP>>) -> test20::Foo<K<TTP>>'}}

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

namespace test25 {

template<typename T, typename...Us>
struct A{
  template<typename V> requires __is_same(V, int)
  A(V);
};

template<typename...TS>
using AA = A<int, TS...>;

template<typename...US>
using BB = AA<US...>; // #test25_BB

BB a{0};
static_assert(__is_same(decltype(a), A<int>));
// FIXME: The template parameter list of generated deduction guide is not strictly conforming,
// as the pack occurs prior to the non-packs.
BB b{0, 1};
// expected-error@-1 {{no viable}}
// expected-note@#test25_BB 2{{not viable}}
// expected-note@#test25_BB {{template <typename ...US, typename V> requires __is_same(V, int) && __is_deducible(test25::AA, test25::A<int, US...>) && __is_deducible(test25::BB, test25::A<int, US...>) BB(V) -> test25::A<int, US...>}}
// expected-note@#test25_BB {{implicit deduction guide}}

}

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

C test{ 42 };
static_assert(__is_same(decltype(test), A<int>));

} // namespace GH125821

namespace GH133132 {

template <class T>
struct A {};

template <class T>
using Foo = A<A<T>>;

template <class T>
using Bar = Foo<T>;

template <class T = int>
using Baz = Bar<T>;

Baz a{};
static_assert(__is_same(decltype(a), A<A<int>>));

} // namespace GH133132

namespace GH131408 {

struct Node {};

template <class T, Node>
struct A {
    A(T) {}
};

template <class T>
using AA = A<T, {}>;

AA a{0};

static_assert(__is_same(decltype(a), A<int, Node{}>));
}

namespace GH130604 {
template <typename T> struct A {
    A(T);
};

template <typename T, template <typename> class TT = A> using Alias = TT<T>; // #gh130604-alias
template <typename T> using Alias2 = Alias<T>;

Alias2 a(42);
// expected-error@-1 {{no viable constructor or deduction guide for deduction of template arguments of 'Alias2'}}
Alias  b(42);
// expected-error@-1 {{alias template 'Alias' requires template arguments; argument deduction only allowed for class templates or alias template}}
// expected-note@#gh130604-alias {{template is declared here}}
}

namespace GH190517 {
template <typename T> struct S1 {};
template <typename T> using S2 = S1<char>;
template <typename T> using S3 = S2<T>; // expected-note {{candidate function not viable}} \
                                        // expected-note {{implicit deduction guide declared}} \
                                        // expected-note {{candidate function not viable}} \
                                        // expected-note {{implicit deduction guide declared}} \
                                        // expected-note {{cannot deduce template arguments for 'GH190517::S3' from 'GH190517::S1<char>'}}
S3 foo(42); // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'S3'}}
}

// Template parameters of the alias template that appear in a synthesized
// deduction guide only through the default template arguments of other
// template parameters cannot be deduced from the constructor arguments. They
// get a default template argument deduced from the return type of the
// underlying deduction guide instead, and the template parameters of the
// synthesized guide are ordered so that default template arguments only refer
// to preceding ones.
namespace synthesized_default_args {
template <class T> struct hash {};
template <class T> struct alloc {};
template <class It> struct iter_traits { using value_type = typename It::value_type; };
struct Iter { using value_type = int; };

template <class Key, class Hash = hash<Key>, class Alloc = alloc<Key>>
struct Set {
  Set();
  template <class It> Set(It, It);
  template <class It> Set(It, It, Hash);
};
template <class It,
          class Hash = hash<typename iter_traits<It>::value_type>,
          class Alloc = alloc<typename iter_traits<It>::value_type>>
Set(It, It, Hash = Hash(), Alloc = Alloc())
    -> Set<typename iter_traits<It>::value_type, Hash, Alloc>;

// Like std::unordered_set: the alias merely renames the class template.
template <class Key, class Hash = hash<Key>, class Alloc = alloc<Key>>
using MySet = Set<Key, Hash, Alloc>;
// The alias has fewer template parameters than the class template.
template <class Key, class Hash = hash<Key>>
using MySet2 = Set<Key, Hash>;
// The alias has a different default template argument, which wins.
template <class Key, class Hash = hash<Key*>, class Alloc = alloc<Key>>
using MySet3 = Set<Key, Hash, Alloc>; // #MySet3

void f(Iter b, Iter e) {
  MySet s1(b, e);
  static_assert(__is_same(decltype(s1), Set<int, hash<int>, alloc<int>>));
  MySet s2(b, e, hash<long>());
  static_assert(__is_same(decltype(s2), Set<int, hash<long>, alloc<int>>));
  MySet2 s3(b, e);
  static_assert(__is_same(decltype(s3), Set<int, hash<int>, alloc<int>>));
  MySet3 s4(b, e);
  static_assert(__is_same(decltype(s4), Set<int, hash<int*>, alloc<int>>));
  MySet s5 = s1;
  static_assert(__is_same(decltype(s5), decltype(s1)));

  // The non-deduced template parameter 'It' of the underlying guide now comes
  // first, followed by 'Key' with its synthesized default template argument.
  MySet3 s6(b, e, 1, 2, 3); // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'MySet3'}}
  // expected-note@#MySet3 {{implicit deduction guide declared as 'template <class It, class Key = typename iter_traits<It>::value_type, class Hash = hash<Key *>, class Alloc = alloc<Key>> requires __is_deducible(synthesized_default_args::MySet3, Set<typename iter_traits<It>::value_type, Hash, Alloc>) MySet3(It, It, Hash, Alloc) -> Set<typename iter_traits<It>::value_type, Hash, Alloc>'}}
  // expected-note@#MySet3 4 {{implicit deduction guide declared as}}
  // expected-note@#MySet3 {{requires at most 4 arguments, but 5 were provided}}
  // expected-note@#MySet3 {{requires 3 arguments, but 5 were provided}}
  // expected-note@#MySet3 {{requires 2 arguments, but 5 were provided}}
  // expected-note@#MySet3 {{requires 1 argument, but 5 were provided}}
  // expected-note@#MySet3 {{requires 0 arguments, but 5 were provided}}
}

// A template parameter of the alias that is deducible from some constructor
// arguments only.
template <class T, class A = alloc<T>> struct Vec {
  template <class It> Vec(It, It);
  template <class It> Vec(It, It, A);
};
template <class It, class A = alloc<typename iter_traits<It>::value_type>>
Vec(It, It, A = A()) -> Vec<typename iter_traits<It>::value_type, A>;
template <class T> using MyVec = Vec<T, hash<T>>;

void g(Iter b, Iter e) {
  MyVec v1(b, e);
  static_assert(__is_same(decltype(v1), Vec<int, hash<int>>));
  MyVec v2(b, e, hash<int>());
  static_assert(__is_same(decltype(v2), Vec<int, hash<int>>));
}

// The underlying guide is constrained, and the alias is a member of a class
// template.
template <class T> concept Any = true;
template <class Key, class Hash = hash<Key>> struct CSet {
  CSet();
  template <class It> CSet(It, It);
};
template <class It, class Hash = hash<typename iter_traits<It>::value_type>>
  requires Any<It> && Any<Hash>
CSet(It, It, Hash = Hash()) -> CSet<typename iter_traits<It>::value_type, Hash>;

template <class U> struct Outer {
  template <class Key, class Hash = hash<Key>> using MyCSet = CSet<Key, Hash>;
};

void h(Iter b, Iter e) {
  Outer<long>::MyCSet s1(b, e);
  static_assert(__is_same(decltype(s1), CSet<int, hash<int>>));
}
template <class U> void h2(Iter b, Iter e) {
  typename Outer<U>::MyCSet s(b, e);
  static_assert(__is_same(decltype(s), CSet<int, hash<int>>));
}
template void h2<char>(Iter, Iter);

// A template template parameter of the alias refers to another template
// parameter of the alias in its own template parameter list, so it must follow
// that one when the template parameters of the synthesized guide are reordered.
template <auto> struct Def {};
template <class Key, class Cmp = hash<Key>, template <Cmp> class TT = Def>
struct TSet {
  template <class It> TSet(It, It, Cmp);
};
template <class It, class Cmp, template <Cmp> class TT = Def>
TSet(It, It, Cmp) -> TSet<typename iter_traits<It>::value_type, Cmp, TT>;
template <class Key, class Cmp = hash<Key>, template <Cmp> class TT = Def>
using MyTSet = TSet<Key, Cmp, TT>; // #MyTSet

void t(Iter b, Iter e) {
  MyTSet s1(b, e, hash<int>());
  static_assert(__is_same(decltype(s1), TSet<int, hash<int>, Def>));

  MyTSet s2(b, e, 1, 2); // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'MyTSet'}}
  // expected-note@#MyTSet {{implicit deduction guide declared as 'template <class It, class Key = typename iter_traits<It>::value_type, class Cmp = hash<Key>, template <Cmp> class TT = Def> requires __is_deducible(synthesized_default_args::MyTSet, TSet<typename iter_traits<It>::value_type, Cmp, TT>) MyTSet(It, It, Cmp) -> TSet<typename iter_traits<It>::value_type, Cmp, TT>'}}
  // expected-note@#MyTSet 2 {{implicit deduction guide declared as}}
  // expected-note@#MyTSet 2 {{requires 3 arguments, but 4 were provided}}
  // expected-note@#MyTSet {{requires 1 argument, but 4 were provided}}
}
} // namespace synthesized_default_args

namespace nttp_deduced_from_alias_in_nondeduced_param_type {
// The template parameter B of the deduction guide of basic_fn is deduced as
// the constant `false` from the alias fn_ref, while the type of the
// non-deduced template parameter `enable_if_t<!bool_constant<B>::value, int>`
// refers to it; this used to crash when rewriting that type for the deduction
// guide of fn_ref.
template <bool B> struct bool_constant { static constexpr bool value = B; };
using false_type = bool_constant<false>;
using true_type = bool_constant<true>;
template <bool, class T = void> struct enable_if {};
template <class T> struct enable_if<true, T> { using type = T; };
template <bool B, class T = void> using enable_if_t = typename enable_if<B, T>::type;

template <class T, class C> struct function {
  template <class U, enable_if_t<!C::value, int> = 0>
  function(U) noexcept {}
};

template <class T> function(T) -> function<T, false_type>;

template <class T, bool B> using basic_fn = function<T, bool_constant<B>>;

template <class T> using fn_ref = basic_fn<T, false>;
template <class T> using fn_disabled = basic_fn<T, true>;
// expected-note@-1 {{candidate template ignored: could not match 'nttp_deduced_from_alias_in_nondeduced_param_type::function<T, bool_constant<true>>' against 'int'}}
// expected-note@-2 {{implicit deduction guide declared as 'template <class T> requires __is_deducible(nttp_deduced_from_alias_in_nondeduced_param_type::basic_fn, nttp_deduced_from_alias_in_nondeduced_param_type::function<T, bool_constant<true>>) && __is_deducible(nttp_deduced_from_alias_in_nondeduced_param_type::fn_disabled, nttp_deduced_from_alias_in_nondeduced_param_type::function<T, bool_constant<true>>) fn_disabled(nttp_deduced_from_alias_in_nondeduced_param_type::function<T, bool_constant<true>>) -> nttp_deduced_from_alias_in_nondeduced_param_type::function<T, bool_constant<true>>'}}
// expected-note@-3 {{candidate template ignored: constraints not satisfied [with T = int]}}
// expected-note@-4 {{cannot deduce template arguments for 'nttp_deduced_from_alias_in_nondeduced_param_type::fn_disabled' from 'function<int, false_type>' (aka 'function<int, bool_constant<false>>')}}
// expected-note@-5 {{implicit deduction guide declared as 'template <class T> requires __is_deducible(nttp_deduced_from_alias_in_nondeduced_param_type::basic_fn, function<T, false_type>) && __is_deducible(nttp_deduced_from_alias_in_nondeduced_param_type::fn_disabled, function<T, false_type>) fn_disabled(T) -> function<T, false_type>'}}

fn_ref f = 0;
static_assert(__is_same(decltype(f), function<int, false_type>));

// The deduction guide derived from the constructor is not formed, as
// substituting B = true into `enable_if_t<!bool_constant<B>::value, int>` fails.
fn_disabled g = 0; // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'fn_disabled'}}

template <class T, int N> struct Arr {
  template <class U, int M = N, enable_if_t<(M > 0), int> = 0>
  Arr(T (&)[M], U = {}) {}
};
template <class T> using Arr3 = Arr<T, 3>;
int arr3[3];
Arr3 a3(arr3, 0);
static_assert(__is_same(decltype(a3), Arr<int, 3>));
} // namespace nttp_deduced_from_alias_in_nondeduced_param_type

namespace lambda_in_alias_rhs {
// The lambda in the RHS of the alias is rewritten in terms of the template
// parameters of the synthesized deduction guide, and must remain dependent
// there, so that it is instantiated along with the guide. Otherwise, the call
// operator of the closure type in the deduced type would keep the parameter
// type T.
template <class T, class F> struct A {
  constexpr A(T t, F f = {}) : v(f(t)) {}
  int v;
};

template <class T> using AA = A<T, decltype([](T x) { return x + 1; })>;
constexpr AA a{41};
static_assert(a.v == 42);

// The lambda comes from the RHS of the alias that AAA is equivalent to.
template <class T> using AAA = AA<T>;
constexpr AAA aa{1};
static_assert(aa.v == 2);

template <class T, class F> struct B {
  B(T, F f = {}) { f({}); }
};

template <class T> using BB = B<T, decltype([](T) {})>;
BB b{0};

} // namespace lambda_in_alias_rhs
