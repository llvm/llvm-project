// RUN: %clang_cc1 -std=c++98 -verify=expected %s
// RUN: %clang_cc1 -std=c++11 -verify=expected %s
// RUN: %clang_cc1 -std=c++14 -verify=expected %s
// RUN: %clang_cc1 -std=c++17 -verify=expected %s
// RUN: %clang_cc1 -std=c++20 -verify=expected,since-cxx20 %s
// RUN: %clang_cc1 -std=c++20 -mlong-double-64 -verify=expected,since-cxx20 %s
// RUN: %clang_cc1 -std=c++23 -verify=expected,since-cxx20,since-cxx23 %s
// RUN: %clang_cc1 -std=c++2c -verify=expected,since-cxx20,since-cxx23,since-cxx26 %s

#if __cplusplus < 202002L
// expected-no-diagnostics
#endif

namespace cwg2819 { // cwg2819: 19 tentatively ready 2023-12-01
#if __cpp_constexpr >= 202306L
  constexpr void* p = nullptr;
  constexpr int* q = static_cast<int*>(p);
  static_assert(q == nullptr);
#endif
}

namespace cwg2847 { // cwg2847: 19 review 2024-03-01

#if __cplusplus >= 202002L

template<typename>
void i();

struct A {
  template<typename>
  void f() requires true;

  template<>
  void f<int>() requires true;
  // since-cxx20-error@-1 {{explicit specialization cannot have a trailing requires clause unless it declares a function template}}

  friend void i<int>() requires true;
  // since-cxx20-error@-1 {{friend specialization cannot have a trailing requires clause unless it declares a function template}}
};

template<typename>
struct B {
  void f() requires true;

  template<typename>
  void g() requires true;

  template<typename>
  void h() requires true;

  template<>
  void h<int>() requires true;
  // since-cxx20-error@-1 {{explicit specialization cannot have a trailing requires clause unless it declares a function template}}

  friend void i<int>() requires true;
  // since-cxx20-error@-1 {{friend specialization cannot have a trailing requires clause unless it declares a function template}}
};

template<>
void B<int>::f() requires true;
// since-cxx20-error@-1 {{explicit specialization cannot have a trailing requires clause unless it declares a function template}}

template<>
template<typename T>
void B<int>::g() requires true;

#endif

} // namespace cwg2847

namespace cwg2851 { // cwg2851: 19

#if __cplusplus >= 202002L
template<typename T, T v> struct Val { static constexpr T value = v; };


// Floating-point promotions

static_assert(Val<long double, 0.0>::value == 0.0L);
static_assert(Val<long double, 0.0f>::value == 0.0L);
static_assert(Val<double, 0.0f>::value == 0.0);
static_assert(Val<long double, -0.0>::value == -0.0L);

static_assert(!__is_same(Val<long double, -0.0>, Val<long double, 0.0L>));
static_assert(__is_same(Val<long double, 0.5>, Val<long double, 0.5L>));

static_assert(__is_same(Val<long double, __builtin_inff()>, Val<long double, __builtin_infl()>));

static_assert(__is_same(Val<long double, __builtin_nanf("")>, Val<long double, static_cast<long double>(__builtin_nanf(""))>));
static_assert(__is_same(Val<long double, __builtin_nansf("")>, Val<long double, static_cast<long double>(__builtin_nansf(""))>));
static_assert(__is_same(Val<long double, __builtin_nanf("0x1")>, Val<long double, static_cast<long double>(__builtin_nanf("0x1"))>));
static_assert(__is_same(Val<long double, __builtin_nansf("0x1")>, Val<long double, static_cast<long double>(__builtin_nansf("0x1"))>));


// Floating-point conversions where the source value can be represented exactly in the destination type

static_assert(Val<float, 0.0L>::value == 0.0L);
static_assert(__is_same(Val<float, 0.0>, Val<float, 0.0L>));
static_assert(__is_same(Val<float, 0.0>, Val<float, 0.0f>));
static_assert(!__is_same(Val<float, -0.0L>, Val<float, 0.0f>));
static_assert(__is_same(Val<float, 0.5L>, Val<float, 0.5f>));
static_assert(__is_same(Val<float, 0.5L>, Val<float, 0.5f>));

static_assert(__is_same(Val<float, double{__FLT_DENORM_MIN__}>, Val<float, __FLT_DENORM_MIN__>));
Val<float, double{__FLT_DENORM_MIN__} / 2.0> _1;
// since-cxx20-error-re@-1 {{non-type template argument evaluates to {{.+}} which cannot be exactly represented in type 'float'}}
Val<float, static_cast<long double>(__FLT_DENORM_MIN__) / 2.0L> _2;
// since-cxx20-error-re@-1 {{non-type template argument evaluates to {{.+}} which cannot be exactly represented in type 'float'}}
Val<float, __DBL_MAX__> _3;
// since-cxx20-error-re@-1 {{non-type template argument evaluates to {{.+}} which cannot be exactly represented in type 'float'}}

static_assert(__is_same(Val<float, __builtin_infl()>, Val<float, __builtin_inff()>));

static_assert(__is_same(Val<float, __builtin_nanl("")>, Val<float, static_cast<float>(__builtin_nanl(""))>));
static_assert(__is_same(Val<float, __builtin_nansl("")>, Val<float, static_cast<float>(__builtin_nansl(""))>));
#if __SIZEOF_LONG_DOUBLE__ > 8
// since-cxx20-error@-2 {{non-type template argument evaluates to nan which cannot be exactly represented in type 'float'}}
#endif
// Payload is shifted right so these payloads will be preserved
static_assert(__is_same(Val<float, __builtin_nan("0xFF00000000")>, Val<float, static_cast<float>(__builtin_nan("0xFF00000000"))>));
static_assert(__is_same(Val<float, __builtin_nans("0xFF00000000")>, Val<float, static_cast<float>(__builtin_nans("0xFF00000000"))>));
static_assert(__is_same(Val<float, __builtin_nanl("0x1")>, Val<float, static_cast<float>(__builtin_nanl("0x1"))>));
// since-cxx20-error@-1 {{non-type template argument evaluates to nan which cannot be exactly represented in type 'float'}}
static_assert(__is_same(Val<float, __builtin_nansl("0x1")>, Val<float, static_cast<float>(__builtin_nansl("0x1"))>));
// since-cxx20-error@-1 {{non-type template argument evaluates to nan which cannot be exactly represented in type 'float'}}
static_assert(__is_same(Val<float, __builtin_nanl("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")>, Val<float, static_cast<float>(__builtin_nanl("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"))>));
// since-cxx20-error@-1 {{non-type template argument evaluates to nan which cannot be exactly represented in type 'float'}}
static_assert(__is_same(Val<float, __builtin_nansl("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")>, Val<float, static_cast<float>(__builtin_nansl("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"))>));
// since-cxx20-error@-1 {{non-type template argument evaluates to nan which cannot be exactly represented in type 'float'}}
#endif

}

namespace cwg2858 { // cwg2858: 19 tentatively ready 2024-04-05

#if __cplusplus > 202302L

template<typename... Ts>
struct A {
  // FIXME: The nested-name-specifier in the following friend declarations are declarative,
  // but we don't treat them as such (yet).
  friend void Ts...[0]::f();
  template<typename U>
  friend void Ts...[0]::g();

  friend struct Ts...[0]::B;
  // FIXME: The index of the pack-index-specifier is printed as a memory address in the diagnostic.
  template<typename U>
  friend struct Ts...[0]::C;
  // expected-warning-re@-1 {{dependent nested name specifier 'Ts...[{{.*}}]::' for friend template declaration is not supported; ignoring this friend declaration}}
};

#endif

} // namespace cwg2858
