// RUN: %check_clang_tidy %s cppcoreguidelines-macro-usage %t \
// RUN: -config="{CheckOptions: \
// RUN:  [{key: cppcoreguidelines-macro-usage.AllowedRegexp, value: 'DEBUG_*|TEST_*'}]}" --

#ifndef INCLUDE_GUARD
#define INCLUDE_GUARD

#define PROBLEMATIC_CONSTANT 0
// CHECK-MESSAGES: [[@LINE-1]]:9: warning: macro 'PROBLEMATIC_CONSTANT' used to declare a constant; consider using a 'constexpr' constant

#define PROBLEMATIC_FUNCTION(x, y) ((a) > (b) ? (a) : (b))
// CHECK-MESSAGES: [[@LINE-1]]:9: warning: function-like macro 'PROBLEMATIC_FUNCTION' used; consider a 'constexpr' template function

#define PROBLEMATIC_VARIADIC(...) (__VA_ARGS__)
// CHECK-MESSAGES: [[@LINE-1]]:9: warning: variadic macro 'PROBLEMATIC_VARIADIC' used; consider using a 'constexpr' variadic template function

#define PROBLEMATIC_VARIADIC2(x, ...) (__VA_ARGS__)
// CHECK-MESSAGES: [[@LINE-1]]:9: warning: variadic macro 'PROBLEMATIC_VARIADIC2' used; consider using a 'constexpr' variadic template function

#define DEBUG_CONSTANT 0
#define DEBUG_FUNCTION(x, y) ((a) > (b) ? (a) : (b))
#define DEBUG_VARIADIC(...) (__VA_ARGS__)
#define TEST_CONSTANT 0
#define TEST_FUNCTION(x, y) ((a) > (b) ? (a) : (b))
#define TEST_VARIADIC(...) (__VA_ARGS__)
#define TEST_VARIADIC2(x, ...) (__VA_ARGS__)

#endif
