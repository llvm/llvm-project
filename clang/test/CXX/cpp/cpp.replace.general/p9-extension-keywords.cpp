// RUN: %clang_cc1 -std=gnu++26 -fms-extensions -fsyntax-only -verify=default %s
// RUN: %clang_cc1 -std=gnu++26 -fms-extensions -pedantic -fsyntax-only -verify=pedantic %s
// RUN: %clang_cc1 -std=gnu++26 -fms-extensions -pedantic-errors -fsyntax-only -verify=pedantic-errors %s

// [cpp.replace.general]/p9 only covers the keywords listed in [lex.key].
// Keywords that are enabled by an extension keep the pedantic-only diagnostic,
// and #undef of them is not diagnosed.

#define typeof decltype
// pedantic-warning@-1 {{keyword is hidden by macro definition}}
// pedantic-errors-error@-2 {{keyword is hidden by macro definition}}
#undef typeof

#define _cdecl
// pedantic-warning@-1 {{keyword is hidden by macro definition}}
// pedantic-errors-error@-2 {{keyword is hidden by macro definition}}
#undef _cdecl

// C keywords that clang accepts in C++ are reserved identifiers and are not
// diagnosed as keywords.
#define _Static_assert static_assert
#undef _Static_assert
#define _Alignof alignof
#undef _Alignof

// Standard C++ keywords are still ill-formed as macro names in GNU mode.
#define decltype typeof
// default-error@-1 {{keyword is hidden by macro definition}}
// pedantic-error@-2 {{keyword is hidden by macro definition}}
// pedantic-errors-error@-3 {{keyword is hidden by macro definition}}
#undef decltype
// pedantic-warning@-1 {{keyword or identifier with special meaning is used as a macro name}}
// pedantic-errors-error@-2 {{keyword or identifier with special meaning is used as a macro name}}

