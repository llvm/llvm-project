// RUN: %clang_cc1 -std=c++26 -pedantic -verify=cxx26,expected -Wno-invalid-pp-token %s
// RUN: %clang_cc1 -std=c++23 -pedantic -verify=cxx23,expected -Wno-invalid-pp-token %s

// P2843R3: Preprocessing is never undefined. The constructs this paper makes
// ill-formed were previously undefined behavior; under C++26 Clang now
// diagnoses them as errors, while retaining the pre-existing pedantic warning
// in earlier language modes for compatibility.

// [cpp.cond] A macro expansion that produces 'defined' in a conditional
// expression. P2843R3 makes this ill-formed; promoted to a hard error in
// C++26.
#define DEFINED defined
// cxx26-error@+2 {{macro expansion producing 'defined' is not allowed}}
// cxx23-warning@+1 {{macro expansion producing 'defined' has undefined behavior}}
#if DEFINED(bar)
#endif

// Malformed 'defined' operands are ill-formed in all modes.
#if defined()      // expected-error {{macro name must be an identifier}}
#endif
#if defined(a b)   // expected-error {{missing ')' after 'defined'}} expected-note {{to match this '('}}
#endif
#if defined(a, b)  // expected-error {{missing ')' after 'defined'}} expected-note {{to match this '('}}
#endif

// [cpp.replace.general] A preprocessing directive inside the arguments of a
// function-like macro invocation. Promoted to a hard error in C++26.
#define FUNCTION_MACRO(...)
// cxx26-note@+1 2 {{expansion of macro 'FUNCTION_MACRO' requested here}}
FUNCTION_MACRO(
    // cxx26-error@+2 {{embedding a #if directive within macro arguments is not supported}}
    // cxx23-warning@+1 {{embedding a directive within macro arguments has undefined behavior}}
    #if 0
    // cxx26-error@+1 {{embedding a #endif directive within macro arguments is not supported}}
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
