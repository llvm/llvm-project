// RUN: %check_clang_tidy %s cppcoreguidelines-macro-usage %t

#ifndef INCLUDE_GUARD
#define INCLUDE_GUARD

#define PROBLEMATIC_CONSTANT 0
// CHECK-MESSAGES: [[@LINE-1]]:9: warning: macro used to declare a constant; consider using a 'constexpr' constant

#define PROBLEMATIC_FUNCTION(x, y) ((a) > (b) ? (a) : (b))
// CHECK-MESSAGES: [[@LINE-1]]:9: warning: function-like macro used; consider a 'constexpr' template function

#define PROBLEMATIC_VARIADIC(...) (__VA_ARGS__)
// CHECK-MESSAGES: [[@LINE-1]]:9: warning: variadic macro used; consider using a 'constexpr' variadic template function

#define PROBLEMATIC_VARIADIC2(x, ...) (__VA_ARGS__)
// CHECK-MESSAGES: [[@LINE-1]]:9: warning: variadic macro used; consider using a 'constexpr' variadic template function

#endif
