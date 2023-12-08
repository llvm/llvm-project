// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -std=c++11 -verify %s
// expected-no-diagnostics

struct true_type {
  static constexpr const bool value = true;
};

struct false_type {
  static constexpr const bool value = false;
};

template <class _Tp, class _Up> struct is_same           : public false_type {};
template <class _Tp>            struct is_same<_Tp, _Tp> : public true_type {};

// Check that our 'is_same' works.
static_assert(is_same<char, char>::value, "is_same is broken");
static_assert(!is_same<char, char *>::value, "is_same is broken");

template <class _Tp, unsigned _AlignOf, unsigned _SizeOf>
struct check_type {
  static constexpr const bool value =
    (alignof(_Tp) == _AlignOf) && (sizeof(_Tp) == _SizeOf);
};

//===----------------------------------------------------------------------===//
// Fundamental types
//===----------------------------------------------------------------------===//

static_assert(check_type<bool, 1, 1>::value, "bool is wrong");

static_assert(check_type<char, 1, 1>::value, "char is wrong");
static_assert(check_type<signed char, 1, 1>::value, "signed char is wrong");
static_assert(check_type<unsigned char, 1, 1>::value, "unsigned char is wrong");

static_assert(check_type<char16_t, 2, 2>::value, "char16_t is wrong");
static_assert(check_type<char32_t, 4, 4>::value, "char32_t is wrong");
static_assert(check_type<wchar_t, 4, 4>::value, "wchar_t is wrong");

static_assert(check_type<short, 2, 2>::value, "short is wrong");
static_assert(check_type<unsigned short, 2, 2>::value, "unsigned short is wrong");

static_assert(check_type<int, 4, 4>::value, "int is wrong");
static_assert(check_type<unsigned int, 4, 4>::value, "unsigned int is wrong");

static_assert(check_type<long, 8, 8>::value, "long is wrong");
static_assert(check_type<unsigned long, 8, 8>::value, "unsigned long is wrong");

static_assert(check_type<long long, 8, 8>::value, "long long is wrong");
static_assert(check_type<unsigned long long, 8, 8>::value, "unsigned long long is wrong");

static_assert(check_type<float, 4, 4>::value, "float is wrong");
static_assert(check_type<double, 8, 8>::value, "double is wrong");
static_assert(check_type<long double, 8, 8>::value, "long double is wrong");

static_assert(check_type<void *, 8, 8>::value, "'void *' is wrong");
static_assert(check_type<int (*)(int), 8, 8>::value, "function pointer is wrong");

//===----------------------------------------------------------------------===//
// stdarg.h
//===----------------------------------------------------------------------===//

#include <stdarg.h>

static_assert(check_type<va_list, 8, 8>::value, "va_list is wrong");

//===----------------------------------------------------------------------===//
// stddef.h
//===----------------------------------------------------------------------===//

#define __STDC_WANT_LIB_EXT1__ 1
#include <stddef.h>

static_assert(is_same<long int, ::ptrdiff_t>::value, "::ptrdiff_t is wrong");
static_assert(is_same<decltype(sizeof(char)), ::size_t>::value, "::size_t is wrong");
static_assert(is_same<long unsigned int, ::size_t>::value, "::size_t is wrong");
static_assert(is_same<long unsigned int, ::rsize_t>::value, "::rsize_t is wrong");
static_assert(is_same<long double, ::max_align_t>::value, "::max_align_t is wrong");

#define __need_wint_t
#include <stddef.h>

static_assert(is_same<int, ::wint_t>::value, "::wint_t is wrong");

