// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify -std=c++17 -Wno-vla-cxx-extension %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify -std=c++20 -Wno-vla-cxx-extension %s

#if !__has_builtin(__builtin_common_reference)
#  error
#endif

// expected-note@*:* {{template <template <class, class, template <class> class, template <class> class> class, template <class ...> class, template <class> class, class, class ...>}}

void test() {
  __builtin_common_reference<> a; // expected-error {{too few template arguments for template '__builtin_common_reference'}}
  __builtin_common_reference<1> b; // expected-error {{template argument for template template parameter must be a class template or type alias template}}
  __builtin_common_reference<int, 1> c; // expected-error {{template argument for template template parameter must be a class template or type alias template}}
}

struct empty_type {};

template <class T>
struct type_identity {
  using type = T;
};

template <class...>
struct common_type;

template <class... Args>
using common_type_t = typename common_type<Args...>::type;

template <class, class, template <class> class, template <class> class>
struct basic_common_reference {};

template <class T, class U, template <class> class TX, template <class> class UX>
using basic_common_reference_t = typename basic_common_reference<T, U, TX, UX>::type;

void test_vla() {
  int i = 4;
  int VLA[i];
  __builtin_common_reference<basic_common_reference_t, common_type_t, type_identity, empty_type, decltype(VLA)> d; // expected-error {{variably modified type 'decltype(VLA)' (aka 'int[i]') cannot be used as a template argument}}
}

template <class... Args>
using common_reference_base = __builtin_common_reference<basic_common_reference_t, common_type_t, type_identity, empty_type, Args...>;

template <class... Args>
struct common_reference : common_reference_base<Args...> {};

template <class... Args>
using common_reference_t = typename __builtin_common_reference<basic_common_reference_t, common_type_t, type_identity, empty_type, Args...>::type;

struct Incomplete;

template<>
struct common_type<Incomplete, Incomplete>;

static_assert(__is_same(common_reference_base<>, empty_type));

static_assert(__is_same(common_reference_base<Incomplete>, type_identity<Incomplete>));
static_assert(__is_same(common_reference_base<char>, type_identity<char>));
static_assert(__is_same(common_reference_base<int>, type_identity<int>));
static_assert(__is_same(common_reference_base<const int>, type_identity<const int>));
static_assert(__is_same(common_reference_base<volatile int>, type_identity<volatile int>));
static_assert(__is_same(common_reference_base<const volatile int>, type_identity<const volatile int>));
static_assert(__is_same(common_reference_base<int[]>, type_identity<int[]>));
static_assert(__is_same(common_reference_base<const int[]>, type_identity<const int[]>));
static_assert(__is_same(common_reference_base<void(&)()>, type_identity<void(&)()>));

static_assert(__is_same(common_reference_base<int[], int[]>, type_identity<int*>));
static_assert(__is_same(common_reference_base<int, int>, type_identity<int>));
static_assert(__is_same(common_reference_base<int, long>, type_identity<long>));
static_assert(__is_same(common_reference_base<long, int>, type_identity<long>));
static_assert(__is_same(common_reference_base<long, long>, type_identity<long>));

static_assert(__is_same(common_reference_base<const int, long>, type_identity<long>));
static_assert(__is_same(common_reference_base<const volatile int, long>, type_identity<long>));
static_assert(__is_same(common_reference_base<int, const long>, type_identity<long>));
static_assert(__is_same(common_reference_base<int, const volatile long>, type_identity<long>));

static_assert(__is_same(common_reference_base<int*, long*>, empty_type));
static_assert(__is_same(common_reference_base<const unsigned int *const &, const unsigned int *const &>, type_identity<const unsigned int *const &>));

static_assert(__is_same(common_reference_base<int, long, float>, type_identity<float>));
static_assert(__is_same(common_reference_base<unsigned, char, long>, type_identity<long>));
static_assert(__is_same(common_reference_base<long long, long long, long>, type_identity<long long>));

static_assert(__is_same(common_reference_base<int [[clang::address_space(1)]]>, type_identity<int [[clang::address_space(1)]]>));
static_assert(__is_same(common_reference_base<int [[clang::address_space(1)]], int>, type_identity<int>));
static_assert(__is_same(common_reference_base<long [[clang::address_space(1)]], int>, type_identity<long>));
static_assert(__is_same(common_reference_base<long [[clang::address_space(1)]], int [[clang::address_space(1)]]>, type_identity<long>));
static_assert(__is_same(common_reference_base<long [[clang::address_space(1)]], long [[clang::address_space(1)]]>, type_identity<long>));
static_assert(__is_same(common_reference_base<long [[clang::address_space(1)]], long [[clang::address_space(2)]]>, type_identity<long>));

struct S {};
struct T : S {};
struct U {};

static_assert(__is_same(common_reference_base<S&&, T&&>, type_identity<S&&>));

static_assert(__is_same(common_reference_base<int S::*, int S::*>, type_identity<int S::*>));
static_assert(__is_same(common_reference_base<int S::*, int T::*>, type_identity<int T::*>));
static_assert(__is_same(common_reference_base<int S::*, long S::*>, empty_type));

static_assert(__is_same(common_reference_base<int (S::*)(), int (S::*)()>, type_identity<int (S::*)()>));
static_assert(__is_same(common_reference_base<int (S::*)(), int (T::*)()>, type_identity<int (T::*)()>));
static_assert(__is_same(common_reference_base<int (S::*)(), long (S::*)()>, empty_type));

static_assert(__is_same(common_reference_base<int&, int&>, type_identity<int&>));
static_assert(__is_same(common_reference_base<int&, const int&>, type_identity<const int&>));
static_assert(__is_same(common_reference_base<volatile int&, const int&>, type_identity<const volatile int&>));

template <class T, class U>
struct my_pair;

template <class T1, class U1, class T2, class U2, template <class> class TX, template <class> class UX>
struct basic_common_reference<my_pair<T1, U1>, my_pair<T2, U2>, TX, UX> {
  using type = my_pair<common_reference_t<TX<T1>, UX<T2>>, common_reference_t<TX<U1>, UX<U2>>>;
};

static_assert(__is_same(common_reference_base<my_pair<const int&, int&>, my_pair<int&, volatile int&>>, type_identity<my_pair<const int&, volatile int&>>));
static_assert(__is_same(common_reference_base<const my_pair<int, int>&, my_pair<int&, volatile int&>>, type_identity<my_pair<const int&, const volatile int&>>));
static_assert(__is_same(common_reference_base<const int&, const volatile int&>, type_identity<const volatile int&>));
static_assert(__is_same(common_reference_base<int&&, const volatile int&>, type_identity<int>));
static_assert(__is_same(common_reference_base<my_pair<int, int>&&, my_pair<int&, volatile int&>>, type_identity<my_pair<const int&, int>>));
static_assert(__is_same(common_reference_base<my_pair<int, int>&&, my_pair<int&, int>&&>, type_identity<my_pair<const int&, int&&>>));

struct conversion_operator {
  operator volatile int&&() volatile;
};

static_assert(__is_same(common_reference_base<volatile conversion_operator&&, volatile int&&>, type_identity<volatile int&&>));

struct reference_wrapper {
  reference_wrapper(int&);
  operator int&() const;
};

static_assert(__is_same(common_reference_base<const reference_wrapper&, int&>, empty_type));
