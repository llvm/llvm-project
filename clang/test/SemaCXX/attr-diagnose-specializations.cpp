// RUN: %clang_cc1 %s -verify

#if !__has_cpp_attribute(clang::diagnose_specializations)
#  error
#endif

struct [[clang::diagnose_specializations]] S {}; // expected-warning {{'diagnose_specializations' attribute only applies to class templates}}

template <class T, class U>
struct [[clang::diagnose_specializations]] is_same {
  static constexpr bool value = __is_same(T, U);
};

template <>
struct is_same<int, char> {}; // expected-error {{'is_same' cannot be specialized}}

template <class>
struct Template {};

template <class T>
struct is_same<Template<T>, Template <T>> {}; // expected-error {{'is_same' cannot be specialized}}

bool test_instantiation1 = is_same<int, int>::value;

template <class T, class U>
[[clang::diagnose_specializations]] inline constexpr bool is_same_v = __is_same(T, U);

template <>
inline constexpr bool is_same_v<int, char> = false; // expected-error {{'is_same_v' cannot be specialized}}

template <class T>
inline constexpr bool is_same_v<Template <T>, Template <T>> = true; // expected-error {{'is_same_v' cannot be specialized}}

bool test_instantiation2 = is_same_v<int, int>;

template <class T>
struct [[clang::diagnose_specializations("specializing type traits results in undefined behaviour")]] is_trivial {
  static constexpr bool value = __is_trivial(T);
};

template <>
struct is_trivial<int> {}; // expected-error {{'is_trivial' cannot be specialized: specializing type traits results in undefined behaviour}}

template <class T>
[[clang::diagnose_specializations("specializing type traits results in undefined behaviour")]] inline constexpr bool is_trivial_v = __is_trivial(T);

template <>
inline constexpr bool is_trivial_v<int> = false; // expected-error {{'is_trivial_v' cannot be specialized: specializing type traits results in undefined behaviour}}
