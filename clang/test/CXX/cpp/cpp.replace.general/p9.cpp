// RUN: %clang_cc1 -std=c++26 %s -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++26 -pedantic-errors -DPEDANTIC_ERROR %s -fsyntax-only -verify

// [cpp.replace.general]/p9: A translation unit shall not #define or #undef
// macro names lexically identical to keywords ([lex.key]), to the identifiers
// listed in Table 4, or to the attribute-tokens described in [dcl.attr], except
// that the macro names likely and unlikely may be defined as function-like
// macros and may be undefined.
#if defined(PEDANTIC_ERROR)
#define for 0    // expected-error {{keyword is hidden by macro definition}}
#undef for       // expected-error {{keyword or identifier with special meaning is used as a macro name}}
#define final 1  // expected-error {{keyword is hidden by macro definition}}
#undef final     // expected-error {{keyword or identifier with special meaning is used as a macro name}}
#define override // expected-error {{keyword is hidden by macro definition}}
#undef override  // expected-error {{keyword or identifier with special meaning is used as a macro name}}
#define const
#undef const     // expected-error {{keyword or identifier with special meaning is used as a macro name}}

#else

#define for 0    // expected-warning {{keyword is hidden by macro definition}}
#undef for       // expected-warning {{keyword or identifier with special meaning is used as a macro name}}
#define final 1  // expected-warning {{keyword is hidden by macro definition}}
#undef final     // expected-warning {{keyword or identifier with special meaning is used as a macro name}}
#define override // expected-warning {{keyword is hidden by macro definition}}
#undef override  // expected-warning {{keyword or identifier with special meaning is used as a macro name}}
#define const
#undef const     // expected-warning {{keyword or identifier with special meaning is used as a macro name}}

#endif

#define assume     // expected-warning {{assume is a reserved attribute identifier}}
#undef assume      // expected-warning {{assume is a reserved attribute identifier}}
#define nodiscard  // expected-warning {{nodiscard is a reserved attribute identifier}}
#undef nodiscard   // expected-warning {{nodiscard is a reserved attribute identifier}}
#define deprecated // expected-warning {{deprecated is a reserved attribute identifier}}
#undef deprecated  // expected-warning {{deprecated is a reserved attribute identifier}}
#define likely     // expected-warning {{likely is a reserved attribute identifier}}
#undef likely      // expected-warning {{likely is a reserved attribute identifier}}
#define unlikely   // expected-warning {{unlikely is a reserved attribute identifier}}
#undef unlikely    // expected-warning {{unlikely is a reserved attribute identifier}}

#define likely(x) (x)
#define unlikely(x) (x)
