// RUN: %clang_cc1 -std=c++26 %s -fsyntax-only -verify

// [cpp.replace.general]/p14: If there are sequences of preprocessing tokens
// within the list of arguments that would otherwise act as preprocessing
// directives, the program is ill-formed.
#define FUNCTION_MACRO(...)
FUNCTION_MACRO(
    #if 0 // expected-warning {{embedding a #if directive within macro arguments has undefined behavior}}
    #endif
)
