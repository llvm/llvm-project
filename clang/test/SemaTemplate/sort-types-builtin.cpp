// RUN: %clang_cc1 -fsyntax-only -verify %s
template <class ...>
struct Types;

// Note that we check for both conditions to ensure our expressions do not accidentally end up being dependent.
static_assert(__is_same(Types<__builtin_sort_types<int, double, int, double>>, Types<double, double, int, int>));
static_assert(!__is_same(Types<__builtin_sort_types<int, double, int, double>>, Types<double, double, int, int>)); // expected-error {{}}

__builtin_sort_types<int, double> err1; // expected-error {{}}
Types<__builtin_sort_types<int, double>*> err2; // expected-error {{}}
Types<const __builtin_sort_types<int, double>> *err3; // expected-error {{}}

template <template <class...> class Inner>
struct Wrapper {
  using result = Inner<int,int,int>*;
};
Types<Wrapper<__builtin_sort_types>::result> *err11; // expected-error  {{use of template '__builtin_sort_types' requires template arguments}} \
                                                     // expected-note@* {{template declaration from hidden source}}



// Check we properly defer evaluations for dependent inputs.
// It is important for performance to avoid repeated sorting (mangling is expensive).
template <class T>
struct CheckDependent {
  template <class U>
  struct Inner {
    using S1 = Types<__builtin_sort_types<T, U>>;
    using S2 = Types<__builtin_sort_types<U, T>>;
    using S3 = Types<__builtin_sort_types<int, T>>;
    using S4 = Types<__builtin_sort_types<U, int>>;
  };
};

using IntDouble = CheckDependent<int>::Inner<double>;
static_assert(__is_same(Types<double, int>, CheckDependent<int>::Inner<double>::S1));
static_assert(!__is_same(Types<double, int>, CheckDependent<int>::Inner<double>::S1));  //expected-error {{}}


// Check that we delay the instantiations.
template <class ...Ts> struct NoInts;
template <> struct NoInts<> {
  static constexpr bool value = true;
};
template <class T, class ...Ts> struct NoInts<T, Ts...> {
  static constexpr bool value = !__is_same(T, int) && NoInts<Ts...>::value;
};
