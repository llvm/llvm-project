// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

static_assert(__has_builtin(__type_pack_element), "");

using SizeT = decltype(sizeof(int));

template <SizeT i, typename ...T>
using TypePackElement = __type_pack_element<i, T...>;

template <int i>
struct X;

static_assert(__is_same(TypePackElement<0, X<0>>, X<0>), "");

static_assert(__is_same(TypePackElement<0, X<0>, X<1>>, X<0>), "");
static_assert(__is_same(TypePackElement<1, X<0>, X<1>>, X<1>), "");

static_assert(__is_same(TypePackElement<0, X<0>, X<1>, X<2>>, X<0>), "");
static_assert(__is_same(TypePackElement<1, X<0>, X<1>, X<2>>, X<1>), "");
static_assert(__is_same(TypePackElement<2, X<0>, X<1>, X<2>>, X<2>), "");

static_assert(__is_same(TypePackElement<0, X<0>, X<1>, X<2>, X<3>>, X<0>), "");
static_assert(__is_same(TypePackElement<1, X<0>, X<1>, X<2>, X<3>>, X<1>), "");
static_assert(__is_same(TypePackElement<2, X<0>, X<1>, X<2>, X<3>>, X<2>), "");
static_assert(__is_same(TypePackElement<3, X<0>, X<1>, X<2>, X<3>>, X<3>), "");

static_assert(__is_same(TypePackElement<0, X<0>, X<1>, X<2>, X<3>, X<4>>, X<0>), "");
static_assert(__is_same(TypePackElement<1, X<0>, X<1>, X<2>, X<3>, X<4>>, X<1>), "");
static_assert(__is_same(TypePackElement<2, X<0>, X<1>, X<2>, X<3>, X<4>>, X<2>), "");
static_assert(__is_same(TypePackElement<3, X<0>, X<1>, X<2>, X<3>, X<4>>, X<3>), "");
static_assert(__is_same(TypePackElement<4, X<0>, X<1>, X<2>, X<3>, X<4>>, X<4>), "");

static_assert(__is_same(TypePackElement<0, X<0>, X<1>, X<2>, X<3>, X<4>, X<5>>, X<0>), "");
static_assert(__is_same(TypePackElement<1, X<0>, X<1>, X<2>, X<3>, X<4>, X<5>>, X<1>), "");
static_assert(__is_same(TypePackElement<2, X<0>, X<1>, X<2>, X<3>, X<4>, X<5>>, X<2>), "");
static_assert(__is_same(TypePackElement<3, X<0>, X<1>, X<2>, X<3>, X<4>, X<5>>, X<3>), "");
static_assert(__is_same(TypePackElement<4, X<0>, X<1>, X<2>, X<3>, X<4>, X<5>>, X<4>), "");
static_assert(__is_same(TypePackElement<5, X<0>, X<1>, X<2>, X<3>, X<4>, X<5>>, X<5>), "");

// Test __type_pack_element with more than 2 top-level template arguments.
static_assert(__is_same(__type_pack_element<5, X<0>, X<1>, X<2>, X<3>, X<4>, X<5>>, X<5>), "");

template <SizeT Index, typename ...T>
using ErrorTypePackElement1 = __type_pack_element<Index, T...>; // expected-error{{may not be accessed at an out of bounds index}}
using illformed1 = ErrorTypePackElement1<3, X<0>, X<1>>;  // expected-note{{in instantiation}}
