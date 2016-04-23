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

#define TEST_CONCAT1(X, Y) X##Y
#define TEST_CONCAT(X, Y) TEST_CONCAT1(X, Y)

#ifdef __has_feature
#define TEST_HAS_FEATURE(X) __has_feature(X)
#else
#define TEST_HAS_FEATURE(X) 0
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

/* Make a nice name for the standard version */
#if  __cplusplus <= 199711L
# define TEST_STD_VER 3
#elif __cplusplus <= 201103L
# define TEST_STD_VER 11
#elif __cplusplus <= 201402L
# define TEST_STD_VER 14
#else
# define TEST_STD_VER 99    // greater than current standard
#endif

/* Features that were introduced in C++11 */
#if TEST_STD_VER >= 11
#define TEST_HAS_RVALUE_REFERENCES
#define TEST_HAS_VARIADIC_TEMPLATES
#define TEST_HAS_INITIALIZER_LISTS
#define TEST_HAS_BASIC_CONSTEXPR
#endif

/* Features that were introduced in C++14 */
#if TEST_STD_VER >= 14
#define TEST_HAS_EXTENDED_CONSTEXPR
#define TEST_HAS_VARIABLE_TEMPLATES
#endif

/* Features that were introduced after C++14 */
#if TEST_STD_VER > 14
#endif

#if TEST_HAS_EXTENSION(cxx_decltype) || TEST_STD_VER >= 11
#define TEST_DECLTYPE(T) decltype(T)
#else
#define TEST_DECLTYPE(T) __typeof__(T)
#endif

#if TEST_STD_VER >= 11
#define TEST_CONSTEXPR constexpr
#define TEST_NOEXCEPT noexcept
# if TEST_STD_VER >= 14
#   define TEST_CONSTEXPR_CXX14 constexpr
# else
#   define TEST_CONSTEXPR_CXX14
# endif
#else
#define TEST_CONSTEXPR
#define TEST_CONSTEXPR_CXX14
#define TEST_NOEXCEPT
#endif

#if TEST_HAS_EXTENSION(cxx_static_assert) || TEST_STD_VER >= 11
#  define TEST_STATIC_ASSERT(Expr, Msg) static_assert(Expr, Msg)
#else
#  define TEST_STATIC_ASSERT(Expr, Msg)                          \
      typedef ::test_detail::static_assert_check<sizeof(         \
          ::test_detail::static_assert_incomplete_test<(Expr)>)> \
    TEST_CONCAT(test_assert, __LINE__)
#
#endif

namespace test_detail {

template <bool> struct static_assert_incomplete_test;
template <> struct static_assert_incomplete_test<true> {};
template <unsigned> struct static_assert_check {};

} // end namespace test_detail


#if !TEST_HAS_FEATURE(cxx_rtti) && !defined(__cxx_rtti)
#define TEST_HAS_NO_RTTI
#endif

#if !TEST_HAS_FEATURE(cxx_exceptions) && !defined(__cxx_exceptions)
#define TEST_HAS_NO_EXCEPTIONS
#endif

#if TEST_HAS_FEATURE(address_sanitizer) || TEST_HAS_FEATURE(memory_sanitizer) || \
    TEST_HAS_FEATURE(thread_sanitizer)
#define TEST_HAS_SANITIZERS
#endif

#endif // SUPPORT_TEST_MACROS_HPP
