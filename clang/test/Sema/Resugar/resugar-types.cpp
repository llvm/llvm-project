// RUN: %clang_cc1 -std=c++2b -fms-extensions -verify %s
// expected-no-diagnostics

static constexpr int alignment = 64; // Suitable large alignment.

struct Baz {};
using Bar [[gnu::aligned(alignment)]] = Baz;
using Int [[gnu::aligned(alignment)]] = int;

#define TEST(X) static_assert(alignof(X) == alignment)
#define TEST_NOT(X) static_assert(alignof(X) != alignment)

// Sanity checks.
TEST_NOT(Baz);
TEST(Bar);

namespace t1 {
template <class T> struct foo { using type = T; };
template <class U> struct foo<U &> { using type = U; };

TEST(typename foo<Bar>::type);
TEST(typename foo<Bar &>::type);
} // namespace t1

namespace t2 {
template <int, class T> struct foo1 { using type = T; };
template <class T> struct foo2 { using type = typename foo1<1, T>::type; };
TEST(typename foo2<Bar>::type);
} // namespace t2

namespace t3 {
template <class T> struct foo1 {
  template <int, class U> struct foo2 { using type1 = T; };
  using type2 = typename foo2<1, int>::type1;
};
TEST(typename foo1<Bar>::type2);
} // namespace t3

namespace t4 {
template <class T> struct foo {
  template <class U> using type1 = T;
  using type2 = type1<int>;
};
TEST(typename foo<Bar>::type2);
} // namespace t4

namespace t5 {
template <class T> struct foo {
  template <int, class U> using type1 = U;
  using type2 = type1<1, T>;
};
TEST(typename foo<Bar>::type2);
} // namespace t5

namespace t6 {
template <class T> struct foo1 {
  template <int, class U> struct foo2 { using type = U; };
  using type2 = typename foo2<1, T>::type;
};
TEST(typename foo1<Bar>::type2);
}; // namespace t6

namespace t7 {
template <class T> struct foo {
  template <int, class U> using type1 = U;
};
using type2 = typename foo<int>::template type1<1, Bar>;
TEST(type2);
} // namespace t7

namespace t8 {
template <class T> struct foo {
  using type1 = T;
};
template <class T, class> using type2 = T;
using type3 = typename type2<foo<Bar>, int>::type1;
TEST(type3);
} // namespace t8

namespace t9 {
template <class A, class B> struct Y {
  using type1 = A;
  using type2 = B;
};
template <class C, class D> using Z = Y<C, D>;
template <class E> struct foo {
  template <class F> using apply = Z<F, E>;
};
using T1 = foo<Bar>::apply<char>;
TEST_NOT(T1::type1);
TEST(T1::type2);

using T2 = foo<int>::apply<Bar>;
TEST(T2::type1);
TEST_NOT(T2::type2);
} // namespace t9

namespace t10 {
template <class A1, class A2> struct Y {
  using type1 = A1;
  using type2 = A2;
};
template <typename... Bs> using Z = Y<Bs...>;
template <typename... Cs> struct foo {
  template <typename... Ds> using bind = Z<Ds..., Cs...>;
};
using T1 = foo<Bar>::bind<char>;
TEST_NOT(T1::type1);
TEST(T1::type2);

using T2 = foo<int>::bind<Bar>;
TEST(T2::type1);
TEST_NOT(T2::type2);
} // namespace t10

namespace t11 {
template <class A1, class A2 = A1> struct A { using type1 = A2; };
TEST(A<Bar>::type1);
} // namespace t11

namespace t12 {
template <class T> struct W {
  template <class Z, template <class Z1, class Z2 = Z1, class Z3 = T> class TT>
  struct X {
    using type1 = TT<Z>;
  };
};

template <class Y1, class Y2, class Y3> struct Y {
  using type2 = Y2;
  using type3 = Y3;
};

using T1 = typename W<Bar>::X<Int, Y>::type1;
TEST(typename T1::type2);
TEST(typename T1::type3);
} // namespace t12

namespace t13 {
template <template <typename...> class C, typename... Us> struct foo {
  template <typename... Ts> using bind = C<Ts..., Us...>;
};
template <typename A1, typename A2> struct Y {
  using type1 = A1;
  using type2 = A2;
};
template <typename... Ts> using Z = Y<Ts...>;

using T1 = typename foo<Z, Bar>::template bind<int>;
TEST_NOT(typename T1::type1);
TEST(typename T1::type2);

using T2 = typename foo<Z, int>::template bind<Bar>;
TEST(typename T2::type1);
TEST_NOT(typename T2::type2);
} // namespace t13

namespace t14 {
template <int, class A1, class...> struct A { using B = A1; };
template <class A2, class A3, class... A4> struct A<0, A2, A3, A4...> {
  using B = typename A<0, A3, A4...>::B;
};
using T1 = typename A<0, long, short, Bar>::B;
TEST(T1);
} // namespace t14

namespace t15 {
template <class T, T... Ints> struct foo { using type1 = T; };
using type2 = typename __make_integer_seq<foo, Int, 1>::type1;
TEST(type2);
template <typename T, T N> using type3 = __make_integer_seq<foo, T, N>;
using type4 = type3<Int, 1>::type1;
TEST(type4);
} // namespace t15

namespace t16 {
template <class A1> struct A {
  using type1 = A1;
};
using type2 = __type_pack_element<0, A<Bar>>::type1;
TEST(type2);
} // namespace t16

namespace t17 {
template <class A1> struct A {
  using type1 = A1;
};
struct C : A<int> {
  using A::type1;
};
TEST(C::A<Bar>::type1);
TEST_NOT(C::A<int>::type1);
TEST(C::A<Int>::type1);
} // namespace t17

namespace t18 {
template <class A1> struct A;
template <class B1> struct B {};
template <class C1> struct A<B<C1>> {
    using type1 = C1;
};
TEST(A<B<Bar>>::type1);
} // namespace t18

namespace t19 {
template <class T> struct A { using type = T&; };

TEST_NOT(A<Bar __unaligned>::type);
} // namespace t19
