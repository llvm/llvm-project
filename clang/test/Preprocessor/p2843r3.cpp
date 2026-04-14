// RUN: %clang_cc1 -std=c++26 -pedantic -verify -Wno-invalid-pp-token %s
// RUN: %clang_cc1 -std=c++23 -pedantic -verify -Wno-invalid-pp-token %s

// P2843R3: Preprocessing is never undefined.
// These constructs were previously "undefined behavior" in the preprocessor;
// as of C++26 they are ill-formed (diagnostic required). Clang already
// diagnoses them under -pedantic, so this test just pins that behavior down.

// [cpp.cond] A macro that expands to 'defined' in a conditional expression.
#define DEFINED defined
#if DEFINED(bar) // expected-warning {{macro expansion producing 'defined' has undefined behavior}}
#endif

// [cpp.replace.general] A preprocessing directive inside the arguments of a
// function-like macro invocation.
#define FUNCTION_MACRO(...)
FUNCTION_MACRO(
    #if 0 // expected-warning {{embedding a directive within macro arguments has undefined behavior}}
    #endif
)

// [cpp.concat] Concatenation that does not form a valid preprocessing token.
#define CONCAT(A, B) A ## B
CONCAT(=, >) // expected-error {{pasting formed '=>', an invalid preprocessing token}}
// expected-error@-1 {{expected unqualified-id}}

// [cpp.predefined] #undef of a reserved identifier / builtin macro.
#undef defined  // expected-error {{'defined' cannot be used as a macro name}}
#undef __DATE__ // expected-warning {{undefining builtin macro}}

// [cpp.line] #line with a non-positive or out-of-range argument.
#line 0          // expected-warning {{#line directive with zero argument is a GNU extension}}
#line -1         // expected-error {{#line directive requires a positive integer argument}}
#line 2147483647 // ok, largest value required to be accepted
#line 2147483648 // expected-warning {{C requires #line number to be less than 2147483648, allowed as extension}}
