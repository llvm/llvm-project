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
struct is_same<int, char> {}; // expected-error {{specializing a template which should not be specialized}}

template <class>
struct Template {};

template <class T>
struct is_same<Template<T>, Template <T>> {}; // expected-error {{specializing a template which should not be specialized}}

bool test_instantiation1 = is_same<int, int>::value;

template <class T, class U>
[[clang::diagnose_specializations]] inline constexpr bool is_same_v = __is_same(T, U);

template <>
inline constexpr bool is_same_v<int, char> = false; // expected-error {{specializing a template which should not be specialized}}

template <class T>
inline constexpr bool is_same_v<Template <T>, Template <T>> = true; // expected-error {{specializing a template which should not be specialized}}

bool test_instantiation2 = is_same_v<int, int>;
