// RUN: %clang_cc1 -fsyntax-only -std=c++20 -triple x86_64-linux-gnu -verify -DEQUIV=1 %s
// RUN: %clang_cc1 -fsyntax-only -std=c++20 -triple x86_64-linux-gnu -verify -DEQUIV=0 -fclang-abi-compat=23 %s
// RUN: %clang_cc1 -fsyntax-only -std=c++20 -triple x86_64-windows-msvc -verify -DEQUIV=0 %s

namespace GH167320 {
template <template <typename...> class, class> struct match {
  static constexpr bool value = false;
};
template <template <typename...> class TT, typename... Ts>
struct match<TT, TT<Ts...>> {
  static constexpr bool value = true;
};
template <typename...> struct my_type {};
template <typename... Ts> using my_type_alias = my_type<Ts...>;

static_assert(match<my_type, my_type<int, int>>::value);
static_assert(match<my_type_alias, my_type<int, int>>::value == EQUIV);
} // namespace GH167320

namespace GH65843 {
template <template <class> class> struct X { static constexpr int value = 1; };
template <class> struct Y {};
template <class T> using Z = Y<T>;
template <> struct X<Y> { static constexpr int value = 2; };

static_assert(__is_same(X<Y>, X<Z>) == EQUIV);
static_assert(X<Z>::value == (EQUIV ? 2 : 1));
} // namespace GH65843

namespace GH177530 {
template <class...> struct tuple {};
template <template <class...> class P> struct A {
  template <class... Args> using packer_alias = P<Args...>;
};
template <class T, template <class...> class OP> struct B {
  static constexpr bool value = false;
};
template <template <class...> class OP, class... Args>
struct B<OP<Args...>, OP> {
  static constexpr bool value = true;
};

static_assert(B<tuple<int, float>, tuple>::value);
static_assert(B<tuple<int, float>, A<tuple>::packer_alias>::value == EQUIV);
} // namespace GH177530

namespace which_aliases {
template <class T, class U> struct P {};
template <class T, class U = int> struct PD {};
template <template <class, class> class> struct X {};

template <class V, class W> using B = P<V, W>;
template <class V, class W> using G = B<V, W>;
static_assert(__is_same(X<P>, X<B>) == EQUIV);
static_assert(__is_same(X<P>, X<G>) == EQUIV);
static_assert(__is_same(X<B>, X<G>) == EQUIV);

template <class V, class W> using C = P<V, V>;
template <class V, class W> using E = P<W, V>;
template <class V, class W> using I = P<V, typename W::type>;
template <class V, class W = int> using F = P<V, W>;
template <class V, class W> using PDA = PD<V, W>;
template <class V, class W> using K = const P<V, W>;
static_assert(!__is_same(X<P>, X<C>));
static_assert(!__is_same(X<P>, X<E>));
static_assert(!__is_same(X<P>, X<I>));
static_assert(!__is_same(X<P>, X<F>));
static_assert(!__is_same(X<PD>, X<PDA>));
static_assert(!__is_same(X<P>, X<K>));

template <class> concept Any = true;
template <template <class V, class W> requires Any<V> class> struct XC {};
template <class V, class W> requires Any<V> using L = P<V, W>;
static_assert(!__is_same(XC<P>, XC<L>));
} // namespace which_aliases

namespace parameter_kinds {
template <int N> struct IntBox {};
template <template <int> class> struct XN {};
template <int M> using IntBoxAlias = IntBox<M>;
static_assert(__is_same(XN<IntBox>, XN<IntBoxAlias>) == EQUIV);

template <template <class> class> struct TTBox {};
template <template <template <class> class> class> struct XT {};
template <template <class> class TT> using TTBoxAlias = TTBox<TT>;
static_assert(__is_same(XT<TTBox>, XT<TTBoxAlias>) == EQUIV);
} // namespace parameter_kinds

namespace deduction_guide {
template <class T> struct DG { DG(T); };
template <class T> using DGAlias = DG<T>;
template <class T> DG(T *) -> DGAlias<T>; // expected-error {{is not written as a specialization of template 'DG'}}
} // namespace deduction_guide
