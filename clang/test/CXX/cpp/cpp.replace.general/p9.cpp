// RUN: %clang_cc1 -std=c++26 -fsyntax-only -verify=default %s
// RUN: %clang_cc1 -std=c++26 -pedantic -fsyntax-only -verify=pedantic %s
// RUN: %clang_cc1 -std=c++26 -pedantic -Wno-keyword-macro-undef -fsyntax-only -verify=default %s
// RUN: %clang_cc1 -std=c++26 -pedantic-errors -fsyntax-only -verify=pedantic-errors %s

// [cpp.replace.general]/p9: A translation unit shall not #define or #undef
// macro names lexically identical to keywords ([lex.key]) or to the identifiers
// listed in Table 4.
#define for 0
// default-warning@-1 {{keyword is hidden by macro definition}}
// pedantic-warning@-2 {{keyword is hidden by macro definition}}
// pedantic-errors-error@-3 {{keyword is hidden by macro definition}}
#undef for
// pedantic-warning@-1 {{keyword or identifier with special meaning is used as a macro name}}
// pedantic-errors-error@-2 {{keyword or identifier with special meaning is used as a macro name}}

#define final 1
// default-warning@-1 {{keyword is hidden by macro definition}}
// pedantic-warning@-2 {{keyword is hidden by macro definition}}
// pedantic-errors-error@-3 {{keyword is hidden by macro definition}}
#undef final
// pedantic-warning@-1 {{keyword or identifier with special meaning is used as a macro name}}
// pedantic-errors-error@-2 {{keyword or identifier with special meaning is used as a macro name}}

#define override
// default-warning@-1 {{keyword is hidden by macro definition}}
// pedantic-warning@-2 {{keyword is hidden by macro definition}}
// pedantic-errors-error@-3 {{keyword is hidden by macro definition}}
#undef override
// pedantic-warning@-1 {{keyword or identifier with special meaning is used as a macro name}}
// pedantic-errors-error@-2 {{keyword or identifier with special meaning is used as a macro name}}

// Empty definitions of qualifier keywords are accepted for compatibility with
// configuration scripts, but #undef is still diagnosed in pedantic modes.
#define const
#undef const
// pedantic-warning@-1 {{keyword or identifier with special meaning is used as a macro name}}
// pedantic-errors-error@-2 {{keyword or identifier with special meaning is used as a macro name}}
