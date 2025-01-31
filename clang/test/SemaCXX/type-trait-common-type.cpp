// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify -std=c++17 -Wno-vla-cxx-extension %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify -std=c++20 -Wno-vla-cxx-extension %s

#if !__has_builtin(__builtin_common_type)
#  error
#endif

// expected-note@*:* {{template parameter from hidden source: template <class ...> class}}
// expected-note@*:* {{template parameter from hidden source: class ...}}
// expected-note@*:* 2{{template parameter from hidden source: template <class ...> class}}

void test() {
  __builtin_common_type<> a; // expected-error {{missing template argument for template parameter}}
  __builtin_common_type<1> b; // expected-error {{template argument for template template parameter must be a class template or type alias template}}
  __builtin_common_type<int, 1> c; // expected-error {{template argument for template template parameter must be a class template or type alias template}}
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

void test_vla() {
  int i = 4;
  int VLA[i];
  __builtin_common_type<common_type_t, type_identity, empty_type, decltype(VLA)> d; // expected-error {{variably modified type 'decltype(VLA)' (aka 'int[i]') cannot be used as a template argument}}
}

template <class... Args>
using common_type_base = __builtin_common_type<common_type_t, type_identity, empty_type, Args...>;

template <class... Args>
struct common_type : common_type_base<Args...> {};

struct Incomplete;

template<>
struct common_type<Incomplete, Incomplete>;

static_assert(__is_same(common_type_base<>, empty_type));
static_assert(__is_same(common_type_base<Incomplete>, empty_type));
static_assert(__is_same(common_type_base<char>, type_identity<char>));
static_assert(__is_same(common_type_base<int>, type_identity<int>));
static_assert(__is_same(common_type_base<const int>, type_identity<int>));
static_assert(__is_same(common_type_base<volatile int>, type_identity<int>));
static_assert(__is_same(common_type_base<const volatile int>, type_identity<int>));
static_assert(__is_same(common_type_base<int[]>, type_identity<int*>));
static_assert(__is_same(common_type_base<const int[]>, type_identity<const int*>));
static_assert(__is_same(common_type_base<void(&)()>, type_identity<void(*)()>));
static_assert(__is_same(common_type_base<int[], int[]>, type_identity<int*>));

static_assert(__is_same(common_type_base<int, int>, type_identity<int>));
static_assert(__is_same(common_type_base<int, long>, type_identity<long>));
static_assert(__is_same(common_type_base<long, int>, type_identity<long>));
static_assert(__is_same(common_type_base<long, long>, type_identity<long>));

static_assert(__is_same(common_type_base<const int, long>, type_identity<long>));
static_assert(__is_same(common_type_base<const volatile int, long>, type_identity<long>));
static_assert(__is_same(common_type_base<int, const long>, type_identity<long>));
static_assert(__is_same(common_type_base<int, const volatile long>, type_identity<long>));

static_assert(__is_same(common_type_base<int*, long*>, empty_type));

static_assert(__is_same(common_type_base<int, long, float>, type_identity<float>));
static_assert(__is_same(common_type_base<unsigned, char, long>, type_identity<long>));
static_assert(__is_same(common_type_base<long long, long long, long>, type_identity<long long>));

static_assert(__is_same(common_type_base<int [[clang::address_space(1)]]>, type_identity<int [[clang::address_space(1)]]>));
static_assert(__is_same(common_type_base<int [[clang::address_space(1)]], int>, type_identity<int>));
static_assert(__is_same(common_type_base<long [[clang::address_space(1)]], int>, type_identity<long>));
static_assert(__is_same(common_type_base<long [[clang::address_space(1)]], int [[clang::address_space(1)]]>, type_identity<long>));
static_assert(__is_same(common_type_base<long [[clang::address_space(1)]], long [[clang::address_space(1)]]>, type_identity<long [[clang::address_space(1)]]>));
static_assert(__is_same(common_type_base<long [[clang::address_space(1)]], long [[clang::address_space(2)]]>, type_identity<long>));

struct S {};
struct T : S {};

static_assert(__is_same(common_type_base<int S::*, int S::*>, type_identity<int S::*>));
static_assert(__is_same(common_type_base<int S::*, int T::*>, type_identity<int T::*>));
static_assert(__is_same(common_type_base<int S::*, long S::*>, empty_type));

static_assert(__is_same(common_type_base<int (S::*)(), int (S::*)()>, type_identity<int (S::*)()>));
static_assert(__is_same(common_type_base<int (S::*)(), int (T::*)()>, type_identity<int (T::*)()>));
static_assert(__is_same(common_type_base<int (S::*)(), long (S::*)()>, empty_type));

struct NoCommonType {};

template <>
struct common_type<NoCommonType, NoCommonType> {};

struct CommonTypeInt {};

template <>
struct common_type<CommonTypeInt, CommonTypeInt> {
  using type = int;
};

template <>
struct common_type<CommonTypeInt, int> {
  using type = int;
};

template <>
struct common_type<int, CommonTypeInt> {
  using type = int;
};

static_assert(__is_same(common_type_base<NoCommonType>, empty_type));
static_assert(__is_same(common_type_base<CommonTypeInt>, type_identity<int>));
static_assert(__is_same(common_type_base<NoCommonType, NoCommonType, NoCommonType>, empty_type));
static_assert(__is_same(common_type_base<CommonTypeInt, CommonTypeInt, CommonTypeInt>, type_identity<int>));
static_assert(__is_same(common_type_base<CommonTypeInt&, CommonTypeInt&&>, type_identity<int>));

static_assert(__is_same(common_type_base<void, int>, empty_type));
static_assert(__is_same(common_type_base<void, void>, type_identity<void>));
static_assert(__is_same(common_type_base<const void, void>, type_identity<void>));
static_assert(__is_same(common_type_base<void, const void>, type_identity<void>));

template <class T>
struct ConvertibleTo {
  operator T();
};

static_assert(__is_same(common_type_base<ConvertibleTo<int>>, type_identity<ConvertibleTo<int>>));
static_assert(__is_same(common_type_base<ConvertibleTo<int>, int>, type_identity<int>));
static_assert(__is_same(common_type_base<ConvertibleTo<int&>, ConvertibleTo<long&>>, type_identity<long>));

struct ConvertibleToB;

struct ConvertibleToA {
  operator ConvertibleToB();
};

struct ConvertibleToB {
  operator ConvertibleToA();
};

static_assert(__is_same(common_type_base<ConvertibleToA, ConvertibleToB>, empty_type));

struct const_ref_convertible {
  operator int&() const &;
  operator int&() && = delete;
};

#if __cplusplus >= 202002L
static_assert(__is_same(common_type_base<const_ref_convertible, int &>, type_identity<int>));
#else
static_assert(__is_same(common_type_base<const_ref_convertible, int &>, empty_type));
#endif

struct WeirdConvertible_1p2_p3 {};

struct WeirdConvertible3 {
  operator WeirdConvertible_1p2_p3();
};

struct WeirdConvertible1p2 {
  operator WeirdConvertible_1p2_p3();
};

template <>
struct common_type<WeirdConvertible3, WeirdConvertible1p2> {
  using type = WeirdConvertible_1p2_p3;
};

template <>
struct common_type<WeirdConvertible1p2, WeirdConvertible3> {
  using type = WeirdConvertible_1p2_p3;
};

struct WeirdConvertible1 {
  operator WeirdConvertible1p2();
};

struct WeirdConvertible2 {
  operator WeirdConvertible1p2();
};

template <>
struct common_type<WeirdConvertible1, WeirdConvertible2> {
  using type = WeirdConvertible1p2;
};

template <>
struct common_type<WeirdConvertible2, WeirdConvertible1> {
  using type = WeirdConvertible1p2;
};

static_assert(__is_same(common_type_base<WeirdConvertible1, WeirdConvertible2, WeirdConvertible3>,
                        type_identity<WeirdConvertible_1p2_p3>));

struct PrivateTypeMember
{
  operator int();
};

template<>
struct common_type<PrivateTypeMember, PrivateTypeMember>
{
private:
  using type = int;
};

static_assert(__is_same(common_type_base<PrivateTypeMember, PrivateTypeMember, PrivateTypeMember>, empty_type));
