// -*- C++ -*-
//===---------------------------- test_macros.h ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TEST_MACROS_HPP
#define SUPPORT_TEST_MACROS_HPP

#include <ciso646> // Get STL specific macros like _LIBCPP_VERSION

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#endif

#define TEST_CONCAT1(X, Y) X##Y
#define TEST_CONCAT(X, Y) TEST_CONCAT1(X, Y)

#ifdef __has_feature
#define TEST_HAS_FEATURE(X) __has_feature(X)
#else
#define TEST_HAS_FEATURE(X) 0
#endif

#ifdef __has_include
#define TEST_HAS_INCLUDE(X) __has_include(X)
#else
#define TEST_HAS_INCLUDE(X) 0
#endif

#ifdef __has_extension
#define TEST_HAS_EXTENSION(X) __has_extension(X)
#else
#define TEST_HAS_EXTENSION(X) 0
#endif

#ifdef __has_builtin
#define TEST_HAS_BUILTIN(X) __has_builtin(X)
#else
#define TEST_HAS_BUILTIN(X) 0
#endif
#ifdef __is_identifier
// '__is_identifier' returns '0' if '__x' is a reserved identifier provided by
// the compiler and '1' otherwise.
#define TEST_HAS_BUILTIN_IDENTIFIER(X) !__is_identifier(X)
#else
#define TEST_HAS_BUILTIN_IDENTIFIER(X) 0
#endif

#if defined(__EDG__)
# define TEST_COMPILER_EDG
#elif defined(__clang__)
# define TEST_COMPILER_CLANG
# if defined(__apple_build_version__)
#  define TEST_COMPILER_APPLE_CLANG
# endif
#elif defined(_MSC_VER)
# define TEST_COMPILER_C1XX
#elif defined(__GNUC__)
# define TEST_COMPILER_GCC
#endif

#if defined(__apple_build_version__)
#define TEST_APPLE_CLANG_VER (__clang_major__ * 100) + __clang_minor__
#elif defined(__clang_major__)
#define TEST_CLANG_VER (__clang_major__ * 100) + __clang_minor__
#elif defined(__GNUC__)
#define TEST_GCC_VER (__GNUC__ * 100 + __GNUC_MINOR__)
#endif

/* Make a nice name for the standard version */
#ifndef TEST_STD_VER
#if  __cplusplus <= 199711L
# define TEST_STD_VER 3
#elif __cplusplus <= 201103L
# define TEST_STD_VER 11
#elif __cplusplus <= 201402L
# define TEST_STD_VER 14
#else
# define TEST_STD_VER 16    // current year; greater than current standard
#endif
#endif

// Attempt to deduce GCC version
#if defined(_LIBCPP_VERSION) && TEST_HAS_INCLUDE(<features.h>)
#include <features.h>
#define TEST_HAS_GLIBC
#define TEST_GLIBC_PREREQ(major, minor) __GLIBC_PREREQ(major, minor)
#endif

/* Features that were introduced in C++14 */
#if TEST_STD_VER >= 14
#define TEST_HAS_EXTENDED_CONSTEXPR
#define TEST_HAS_VARIABLE_TEMPLATES
#endif

/* Features that were introduced after C++14 */
#if TEST_STD_VER > 14
#endif

#if TEST_STD_VER >= 11
#define TEST_ALIGNOF(...) alignof(__VA_ARGS__)
#define TEST_ALIGNAS(...) alignas(__VA_ARGS__)
#define TEST_CONSTEXPR constexpr
#define TEST_NOEXCEPT noexcept
#define TEST_NOEXCEPT_FALSE noexcept(false)
#define TEST_NOEXCEPT_COND(...) noexcept(__VA_ARGS__)
# if TEST_STD_VER >= 14
#   define TEST_CONSTEXPR_CXX14 constexpr
# else
#   define TEST_CONSTEXPR_CXX14
# endif
# if TEST_STD_VER > 14
#   define TEST_THROW_SPEC(...)
# else
#   define TEST_THROW_SPEC(...) throw(__VA_ARGS__)
# endif
#else
#define TEST_ALIGNOF(...) __alignof(__VA_ARGS__)
#define TEST_ALIGNAS(...) __attribute__((__aligned__(__VA_ARGS__)))
#define TEST_CONSTEXPR
#define TEST_CONSTEXPR_CXX14
#define TEST_NOEXCEPT throw()
#define TEST_NOEXCEPT_FALSE
#define TEST_NOEXCEPT_COND(...)
#define TEST_THROW_SPEC(...) throw(__VA_ARGS__)
#endif

#define TEST_ALIGNAS_TYPE(...) TEST_ALIGNAS(TEST_ALIGNOF(__VA_ARGS__))

#if !TEST_HAS_FEATURE(cxx_rtti) && !defined(__cpp_rtti) \
    && !defined(__GXX_RTTI)
#define TEST_HAS_NO_RTTI
#endif

#if !TEST_HAS_FEATURE(cxx_exceptions) && !defined(__cpp_exceptions) \
     && !defined(__EXCEPTIONS)
#define TEST_HAS_NO_EXCEPTIONS
#endif

#if TEST_HAS_FEATURE(address_sanitizer) || TEST_HAS_FEATURE(memory_sanitizer) || \
    TEST_HAS_FEATURE(thread_sanitizer)
#define TEST_HAS_SANITIZERS
#endif

#if defined(_LIBCPP_NORETURN)
#define TEST_NORETURN _LIBCPP_NORETURN
#else
#define TEST_NORETURN [[noreturn]]
#endif

#if TEST_STD_VER < 11
#define ASSERT_NOEXCEPT(...)
#define ASSERT_NOT_NOEXCEPT(...)
#else
#define ASSERT_NOEXCEPT(...) \
    static_assert(noexcept(__VA_ARGS__), "Operation must be noexcept")

#define ASSERT_NOT_NOEXCEPT(...) \
    static_assert(!noexcept(__VA_ARGS__), "Operation must NOT be noexcept")
#endif

/* Macros for testing libc++ specific behavior and extensions */
#if defined(_LIBCPP_VERSION)
#define LIBCPP_ASSERT(...) assert(__VA_ARGS__)
#define LIBCPP_STATIC_ASSERT(...) static_assert(__VA_ARGS__)
#define LIBCPP_ASSERT_NOEXCEPT(...) ASSERT_NOEXCEPT(__VA_ARGS__)
#define LIBCPP_ASSERT_NOT_NOEXCEPT(...) ASSERT_NOT_NOEXCEPT(__VA_ARGS__)
#define LIBCPP_ONLY(...) __VA_ARGS__
#else
#define LIBCPP_ASSERT(...) ((void)0)
#define LIBCPP_STATIC_ASSERT(...) ((void)0)
#define LIBCPP_ASSERT_NOEXCEPT(...) ((void)0)
#define LIBCPP_ASSERT_NOT_NOEXCEPT(...) ((void)0)
#define LIBCPP_ONLY(...) ((void)0)
#endif

namespace test_macros_detail {
template <class T, class U>
struct is_same { enum { value = 0};} ;
template <class T>
struct is_same<T, T> { enum {value = 1}; };
} // namespace test_macros_detail

#define ASSERT_SAME_TYPE(...) \
    static_assert((test_macros_detail::is_same<__VA_ARGS__>::value), \
                 "Types differ uexpectedly")

#ifndef TEST_HAS_NO_EXCEPTIONS
#define TEST_THROW(...) throw __VA_ARGS__
#else
#if defined(__GNUC__)
#define TEST_THROW(...) __builtin_abort()
#else
#include <stdlib.h>
#define TEST_THROW(...) ::abort()
#endif
#endif

#if defined(__GNUC__) || defined(__clang__)
template <class Tp>
inline void DoNotOptimize(Tp const& value) {
  asm volatile("" : : "g"(value) : "memory");
}
#else
#include <intrin.h>
template <class Tp>
inline void DoNotOptimize(Tp const& value) {
  const volatile void* volatile unused = __builtin_addressof(value);
  static_cast<void>(unused);
  _ReadWriteBarrier();
}
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#endif // SUPPORT_TEST_MACROS_HPP
