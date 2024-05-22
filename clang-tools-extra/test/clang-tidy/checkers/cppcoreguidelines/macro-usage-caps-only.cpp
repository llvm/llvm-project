// RUN: %check_clang_tidy %s cppcoreguidelines-macro-usage %t \
// RUN: -config='{CheckOptions: \
// RUN:  {cppcoreguidelines-macro-usage.CheckCapsOnly: true}}' --

#ifndef INCLUDE_GUARD
#define INCLUDE_GUARD

#define problematic_constant 0
// CHECK-MESSAGES: [[@LINE-1]]:9: warning: macro definition does not define the macro name 'problematic_constant' using all uppercase characters

#define problematic_function(x, y) ((a) > (b) ? (a) : (b))
// CHECK-MESSAGES: [[@LINE-1]]:9: warning: macro definition does not define the macro name 'problematic_function' using all uppercase characters

#define problematic_variadic(...) (__VA_ARGS__)
// CHECK-MESSAGES: [[@LINE-1]]:9: warning: macro definition does not define the macro name 'problematic_variadic' using all uppercase characters
//
#define problematic_variadic2(x, ...) (__VA_ARGS__)
// CHECK-MESSAGES: [[@LINE-1]]:9: warning: macro definition does not define the macro name 'problematic_variadic2' using all uppercase characters

#define OKISH_CONSTANT 42
#define OKISH_FUNCTION(x, y) ((a) > (b) ? (a) : (b))
#define OKISH_VARIADIC(...) (__VA_ARGS__)

#endif
