// RUN: %clang_cc1 -fsyntax-only -fms-extensions -verify -verify-ignore-unexpected=note %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -fms-extensions -verify -verify-ignore-unexpected=note %s

// Use -fsyntax-only so pragma handlers can produce annotation tokens.
#define M _Pragma("pack(push, 1)")

#if (0 + M 1) // expected-error {{invalid token at start of a preprocessor expression}}
#endif

#if _Pragma("pack(push, 1)") 1 // expected-error {{invalid token at start of a preprocessor expression}}
#endif

#if 0
#elif M 1 // expected-error {{invalid token at start of a preprocessor expression}}
#endif

#if !M 1 // expected-error {{invalid token at start of a preprocessor expression}}
#endif

#if 0 && M 1 // expected-error {{invalid token at start of a preprocessor expression}}
#endif

#if (0 + __pragma(pack(push, 1)) 1) // expected-error {{invalid token at start of a preprocessor expression}}
#endif
