// RUN: %clang_cc1 -fsyntax-only -Wno-c++11-narrowing -Wno-literal-conversion -std=c++20 -verify %s
// expected-no-diagnostics
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
using Bar = Foo<X, sizeof(X)>;

// FIXME: we should reject this case? GCC rejects it, MSVC accepts it.
Bar s = {{1}};
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
// FIXME: we have an out-bound crash on instantating the synthesized deduction guide `auto (B<C2>) -> B<C2>`
// where C2 should be at the index 0, however, it is still refers the original one where index is 1
template<class X, class Y=A> using AFoo = Foo<Y>;

AFoo s = {1};
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
} // namespace test13

namespace test14 {
template <class T> struct Foo { Foo(T); };

template<class V> using AFoo = Foo<V *>;
template<typename> concept False = false;
template<False W> using BFoo = AFoo<W>;
int i = 0;
AFoo a1(&i); // OK, deduce Foo<int *>

// FIXME: we should reject this case as the W is not deduced from the deduced
// type Foo<int *>.
BFoo b2(&i); 
} // namespace test14
