// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fsyntax-only -verify -DITANIUM %s
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -std=c++20 -fsyntax-only -verify -DMICROSOFT %s

template <class...> struct TypeList;

template <class... Ts>
struct Sorted {
  using type = TypeList<__builtin_sort_pack<Ts...>...>;
};

template <class... Ts>
struct UniqueSorted {
  using type = TypeList<__builtin_sort_pack<__builtin_dedup_pack<Ts...>...>...>;
};

template <class... Ts>
struct SortTwice {
  using Once = TypeList<__builtin_sort_pack<Ts...>...>;
  using Twice = TypeList<__builtin_sort_pack<__builtin_sort_pack<Ts...>...>...>;
};

namespace std {
struct strong_ordering {
  enum __order { LT = -1, EQ = 0, GT = 1 };
  __order value;

  constexpr explicit strong_ordering(__order value) : value(value) {}
  constexpr bool operator==(strong_ordering const &other) const {
    return value == other.value;
  }
  static const strong_ordering less;
  static const strong_ordering equal;
  static const strong_ordering greater;
};

inline constexpr strong_ordering strong_ordering::less(__order::LT);
inline constexpr strong_ordering strong_ordering::equal(__order::EQ);
inline constexpr strong_ordering strong_ordering::greater(__order::GT);
} // namespace std

struct A {};
struct B {};

static_assert(__is_same(Sorted<>::type, TypeList<>));
static_assert(__is_same(Sorted<int>::type, TypeList<int>));
static_assert(__is_same(Sorted<int, int, int>::type, TypeList<int, int, int>));
static_assert(__is_same(Sorted<A, B>::type, TypeList<A, B>));
static_assert(__is_same(Sorted<B, A>::type, TypeList<A, B>));

#ifdef ITANIUM
static_assert(__is_same(Sorted<int, double>::type, TypeList<double, int>));
static_assert(__is_same(Sorted<int, double, int, double>::type,
                        TypeList<double, double, int, int>));
static_assert(__is_same(Sorted<char, int, double, float>::type,
                        TypeList<char, double, float, int>));
static_assert(__is_same(Sorted<A, int>::type, TypeList<A, int>));
static_assert(__is_same(UniqueSorted<int, double, int, float, double>::type,
                        TypeList<double, float, int>));
#endif

#ifdef MICROSOFT
static_assert(__is_same(Sorted<int, double>::type, TypeList<int, double>));
static_assert(__is_same(Sorted<int, double, int, double>::type,
                        TypeList<int, int, double, double>));
static_assert(__is_same(Sorted<char, int, double, float>::type,
                        TypeList<char, int, float, double>));
static_assert(__is_same(Sorted<A, int>::type, TypeList<int, A>));
static_assert(__is_same(UniqueSorted<int, double, int, float, double>::type,
                        TypeList<int, float, double>));
#endif

using Int = int;
using Dbl = double;
static_assert(__is_same(Sorted<Int, Dbl>::type, Sorted<int, double>::type));
static_assert(__is_same(UniqueSorted<int, int, int>::type, TypeList<int>));

static_assert(__is_same(SortTwice<B, int, A, double>::Once,
                        SortTwice<B, int, A, double>::Twice));

template <class T, class U>
struct SortMatchesTypeOrder {
  using SortedTU = TypeList<__builtin_sort_pack<T, U>...>;
  using SortedUT = TypeList<__builtin_sort_pack<U, T>...>;
  static constexpr auto Cmp = __builtin_type_order(T, U);
  static_assert(Cmp != std::strong_ordering::greater
                    ? __is_same(SortedTU, TypeList<T, U>)
                    : __is_same(SortedTU, TypeList<U, T>));
  static_assert(__is_same(SortedTU, SortedUT));
};

template struct SortMatchesTypeOrder<int, int>;
template struct SortMatchesTypeOrder<int, double>;
template struct SortMatchesTypeOrder<double, int>;
template struct SortMatchesTypeOrder<A, B>;
template struct SortMatchesTypeOrder<B, A>;
template struct SortMatchesTypeOrder<int, const int>;
template struct SortMatchesTypeOrder<const int, int>;
template struct SortMatchesTypeOrder<A, int>;
template struct SortMatchesTypeOrder<void *, const void *>;
template struct SortMatchesTypeOrder<int *, int[]>;

template <class A, class B, class C>
struct AdjacentPairsSorted {
  using T0 = __type_pack_element<0, __builtin_sort_pack<A, B, C>...>;
  using T1 = __type_pack_element<1, __builtin_sort_pack<A, B, C>...>;
  using T2 = __type_pack_element<2, __builtin_sort_pack<A, B, C>...>;
  static_assert(__builtin_type_order(T0, T1) != std::strong_ordering::greater);
  static_assert(__builtin_type_order(T1, T2) != std::strong_ordering::greater);
};

template struct AdjacentPairsSorted<int, double, char>;
template struct AdjacentPairsSorted<B, A, int>;
template struct AdjacentPairsSorted<const int, int, volatile int>;

template <class T, class U>
struct Dependent {
  using S1 = TypeList<__builtin_sort_pack<T, U>...>;
  using S2 = TypeList<__builtin_sort_pack<U, T>...>;
  using S3 = TypeList<__builtin_sort_pack<double, T>...>;
  using S4 = TypeList<__builtin_sort_pack<U, int>...>;
};

static_assert(__is_same(Dependent<int, double>::S1, Dependent<int, double>::S2));
static_assert(__is_same(Dependent<int, double>::S1, Sorted<int, double>::type));
static_assert(__is_same(Dependent<int, double>::S3, Sorted<double, int>::type));
static_assert(__is_same(Dependent<int, double>::S4, Sorted<double, int>::type));

template <class... Ts>
struct DependentPack {
  using type = TypeList<__builtin_sort_pack<Ts...>...>;
};

static_assert(__is_same(DependentPack<>::type, TypeList<>));
static_assert(__is_same(DependentPack<B, A, B>::type, Sorted<B, A, B>::type));

__builtin_sort_pack<int, double> err1; // expected-error {{cannot be used outside of template}} \
                                       // expected-error {{declaration type contains an unexpanded parameter pack}}
TypeList<__builtin_sort_pack<int, double> *> err2; // expected-error {{cannot be used outside of template}} \
                                                   // expected-error {{declaration type contains an unexpanded parameter pack}}
TypeList<const __builtin_sort_pack<int, double>> *err3; // expected-error {{cannot be used outside of template}} \
                                                        // expected-error {{declaration type contains an unexpanded parameter pack}}

template <template <class...> class Inner>
struct Wrapper {
  using result = Inner<int, int, int> *;
};
TypeList<Wrapper<__builtin_sort_pack>::result> *err11; // expected-error {{cannot be used outside of template}} \
                                                       // expected-error {{use of template '__builtin_sort_pack' requires template arguments}} \
                                                       // expected-note@* {{template declaration from hidden source}}

template <template <class...> class T = __builtin_sort_pack> // expected-error {{use of template '__builtin_sort_pack' requires template arguments}} \
                                                             // expected-note@* {{template declaration from hidden source}}
struct UseAsTemplate;

static_assert(__is_same(TypeList<__builtin_sort_pack<int>...>, TypeList<int>)); // expected-error {{outside}}

template <class>
struct UnexpandedInTemplate {
  static_assert(__is_same( // expected-error {{static assertion contains an unexpanded parameter pack}}
      TypeList<__builtin_sort_pack<int, double>>, TypeList<double, int>));
};
