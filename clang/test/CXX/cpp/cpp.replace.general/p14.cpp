// RUN: %clang_cc1 -std=c++26 %s -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++26 -pedantic-errors -DPEDANTIC_ERROR %s -fsyntax-only -verify

// [cpp.replace.general]/p14: If there are sequences of preprocessing tokens
// within the list of arguments that would otherwise act as preprocessing
// directives, the program is ill-formed.
#define FUNCTION_MACRO(...)
#if defined(PEDANTIC_ERROR)
FUNCTION_MACRO(
    #if 0 // expected-error {{embedding a #if directive within macro arguments has undefined behavior}}
    #endif
)
FUNCTION_MACRO(
    # // expected-error {{embedding a # directive within macro arguments has undefined behavior}}
)
#else
FUNCTION_MACRO(
    #if 0 // expected-warning {{embedding a #if directive within macro arguments has undefined behavior}}
    #endif
)
FUNCTION_MACRO(
    #  // expected-warning {{embedding a # directive within macro arguments has undefined behavior}}
)
#endif
