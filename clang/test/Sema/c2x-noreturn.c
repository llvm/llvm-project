// RUN: %clang_cc1 -verify=all,c2x -std=c2x -fsyntax-only %s
// RUN: %clang_cc1 -verify=all -std=c17 -fdouble-square-bracket-attributes -fsyntax-only %s
// RUN: %clang_cc1 -verify=none -Wno-deprecated-attributes -D_CLANG_DISABLE_CRT_DEPRECATION_WARNINGS -std=c2x -fsyntax-only %s
// RUN: %clang_cc1 -verify=none -Wno-deprecated-attributes -D_CLANG_DISABLE_CRT_DEPRECATION_WARNINGS -std=c17 -fdouble-square-bracket-attributes -fsyntax-only %s
// none-no-diagnostics

// Test preprocessor functionality.
#if !__has_c_attribute(noreturn)
#error "No noreturn attribute support?"
#endif

#if !__has_c_attribute(_Noreturn)
#error "No _Noreturn attribute support?"
#endif

#if __has_c_attribute(noreturn) != __has_c_attribute(_Noreturn) || \
    __has_c_attribute(noreturn) != 202202L
#error "Wrong value for __has_c_attribute(noreturn)"
#endif

// If we're testings with deprecations disabled, we don't care about testing
// the scenarios that trigger errors because we're only interested in the
// deprecation warning behaviors.
#ifndef _CLANG_DISABLE_CRT_DEPRECATION_WARNINGS
// Test that the attribute accepts no args, applies to the correct subject, etc.
[[noreturn(12)]] void func4(void); // all-error {{attribute 'noreturn' cannot have an argument list}}
[[noreturn]] int not_a_func; // all-error {{'noreturn' attribute only applies to functions}}
void func5(void) [[noreturn]]; // all-error {{'noreturn' attribute cannot be applied to types}}
#endif

_Noreturn void func1(void); // ok, using the function specifier
[[noreturn]] void func2(void);

// This is deprecated because it's only for compatibility with inclusion of the
// <stdnoreturn.h> header where the noreturn macro expands to _Noreturn.
[[_Noreturn]] void func3(void); // all-warning {{the '[[_Noreturn]]' attribute spelling is deprecated in C2x; use '[[noreturn]]' instead}}

// Test the behavior of including <stdnoreturn.h>
#include <stdnoreturn.h> // c2x-warning@stdnoreturn.h:* {{the '<stdnoreturn.h>' header is deprecated in C2x}}

[[noreturn]] void func6(void); // all-warning {{the '[[_Noreturn]]' attribute spelling is deprecated in C2x; use '[[noreturn]]' instead}} \
                               // c2x-warning {{macro 'noreturn' has been marked as deprecated}} \
                               // c2x-note@stdnoreturn.h:* {{macro marked 'deprecated' here}}

void func7 [[noreturn]] (void); // all-warning {{the '[[_Noreturn]]' attribute spelling is deprecated in C2x; use '[[noreturn]]' instead}} \
                                // c2x-warning {{macro 'noreturn' has been marked as deprecated}} \
                                // c2x-note@stdnoreturn.h:* {{macro marked 'deprecated' here}}

noreturn void func8(void); // c2x-warning {{macro 'noreturn' has been marked as deprecated}} \
                           // c2x-note@stdnoreturn.h:* {{macro marked 'deprecated' here}}

// Ensure the function specifier form still works
void noreturn func9(void); // c2x-warning {{macro 'noreturn' has been marked as deprecated}} \
                           // c2x-note@stdnoreturn.h:* {{macro marked 'deprecated' here}}

// Test preprocessor functionality after including <stdnoreturn.h>.
#if !__has_c_attribute(noreturn) // c2x-warning {{macro 'noreturn' has been marked as deprecated}} \
                                 // c2x-note@stdnoreturn.h:* {{macro marked 'deprecated' here}}
#error "No noreturn attribute support?"
#endif

#if !__has_c_attribute(_Noreturn)
#error "No _Noreturn attribute support?"
#endif
