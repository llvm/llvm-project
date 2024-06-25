// RUN: %check_clang_tidy %s cppcoreguidelines-macro-usage -std=c++17-or-later %t -- -header-filter=.* -system-headers --

#ifndef INCLUDE_GUARD
#define INCLUDE_GUARD

#define PROBLEMATIC_CONSTANT 0
// CHECK-MESSAGES: [[@LINE-1]]:9: warning: macro 'PROBLEMATIC_CONSTANT' used to declare a constant; consider using a 'constexpr' constant

#define PROBLEMATIC_CONSTANT_CHAR '0'
// CHECK-MESSAGES: [[@LINE-1]]:9: warning: macro 'PROBLEMATIC_CONSTANT_CHAR' used to declare a constant; consider using a 'constexpr' constant

#define PROBLEMATIC_CONSTANT_WIDE_CHAR L'0'
// CHECK-MESSAGES: [[@LINE-1]]:9: warning: macro 'PROBLEMATIC_CONSTANT_WIDE_CHAR' used to declare a constant; consider using a 'constexpr' constant

#define PROBLEMATIC_CONSTANT_UTF8_CHAR u8'0'
// CHECK-MESSAGES: [[@LINE-1]]:9: warning: macro 'PROBLEMATIC_CONSTANT_UTF8_CHAR' used to declare a constant; consider using a 'constexpr' constant

#define PROBLEMATIC_CONSTANT_UTF16_CHAR u'0'
// CHECK-MESSAGES: [[@LINE-1]]:9: warning: macro 'PROBLEMATIC_CONSTANT_UTF16_CHAR' used to declare a constant; consider using a 'constexpr' constant

#define PROBLEMATIC_CONSTANT_UTF32_CHAR U'0'
// CHECK-MESSAGES: [[@LINE-1]]:9: warning: macro 'PROBLEMATIC_CONSTANT_UTF32_CHAR' used to declare a constant; consider using a 'constexpr' constant

#define PROBLEMATIC_FUNCTION(x, y) ((a) > (b) ? (a) : (b))
// CHECK-MESSAGES: [[@LINE-1]]:9: warning: function-like macro 'PROBLEMATIC_FUNCTION' used; consider a 'constexpr' template function

#define PROBLEMATIC_VARIADIC(...) (__VA_ARGS__)
// CHECK-MESSAGES: [[@LINE-1]]:9: warning: variadic macro 'PROBLEMATIC_VARIADIC' used; consider using a 'constexpr' variadic template function

#define PROBLEMATIC_VARIADIC2(x, ...) (__VA_ARGS__)
// CHECK-MESSAGES: [[@LINE-1]]:9: warning: variadic macro 'PROBLEMATIC_VARIADIC2' used; consider using a 'constexpr' variadic template function

// These are all examples of common macros that shouldn't have constexpr suggestions.
#define CONCAT_NAME(a, b) a##b

#define CONCAT_STR(a, b) #a #b

#define COMMA ,

#define NORETURN [[noreturn]]

#define DEPRECATED attribute((deprecated))

#if LIB_EXPORTS
#define DLLEXPORTS __declspec(dllexport)
#else
#define DLLEXPORTS __declspec(dllimport)
#endif

#endif
