// RUN: %clang_cc1 %s -verify

#if !__has_cpp_attribute(clang::no_specializations)
#  error
#endif

struct [[clang::no_specializations]] S {}; // expected-warning {{'no_specializations' attribute only applies to class templates, function templates, and variable templates}}

template <class T, class U>
struct [[clang::no_specializations]] is_same { // expected-note 2 {{marked 'no_specializations' here}}
  static constexpr bool value = __is_same(T, U);
};

template <class T>
using alias [[clang::no_specializations]] = T; // expected-warning {{'no_specializations' attribute only applies to class templates, function templates, and variable templates}}

template <>
struct is_same<int, char> {}; // expected-error {{'is_same' cannot be specialized}}

template <class>
struct Template {};

template <class T>
struct is_same<Template<T>, Template <T>> {}; // expected-error {{'is_same' cannot be specialized}}

bool test_instantiation1 = is_same<int, int>::value;

template <class T, class U>
[[clang::no_specializations]] inline constexpr bool is_same_v = __is_same(T, U); // expected-note 2 {{marked 'no_specializations' here}}

template <>
inline constexpr bool is_same_v<int, char> = false; // expected-error {{'is_same_v' cannot be specialized}}

template <class T>
inline constexpr bool is_same_v<Template <T>, Template <T>> = true; // expected-error {{'is_same_v' cannot be specialized}}

bool test_instantiation2 = is_same_v<int, int>;

template <class T>
struct [[clang::no_specializations("specializing type traits results in undefined behaviour")]] is_trivial { // expected-note {{marked 'no_specializations' here}}
  static constexpr bool value = __is_trivial(T);
};

template <>
struct is_trivial<int> {}; // expected-error {{'is_trivial' cannot be specialized: specializing type traits results in undefined behaviour}}

template <class T>
[[clang::no_specializations("specializing type traits results in undefined behaviour")]] inline constexpr bool is_trivial_v = __is_trivial(T); // expected-note {{marked 'no_specializations' here}}

template <>
inline constexpr bool is_trivial_v<int> = false; // expected-error {{'is_trivial_v' cannot be specialized: specializing type traits results in undefined behaviour}}

template <class T>
struct Partial {};

template <class T>
struct [[clang::no_specializations]] Partial<Template <T>> {}; // expected-warning {{'no_specializations' attribute only applies to class templates, function templates, and variable templates}}

template <class T>
[[clang::no_specializations]] void func(); // expected-note {{marked 'no_specializations' here}}

template <> void func<int>(); // expected-error {{'func' cannot be specialized}}
