// RUN: %clang_cc1 -std=c++26 -freflection -fsyntax-only %s -verify=metabuiltin
// RUN: %clang_cc1 -std=c++26 -fsyntax-only %s -verify=nometabuiltin
// RUN: %clang_cc1 -std=c++23 -fsyntax-only %s -verify=nometabuiltin
// RUN: %clang_cc1 -x c -std=c23 -fsyntax-only %s -verify=nometabuiltin

// Metafunction builtins are only enabled in C++26 (-freflection enabled).

#if __has_builtin(__builtin_meta_is_type)
#error has the builtin
#else
#error does not have the builtin
#endif
// metabuiltin-error@-4 {{has the builtin}}
// nometabuiltin-error@-3 {{does not have the builtin}}

#if __has_constexpr_builtin(__builtin_meta_is_type)
#error has the constexpr builtin
#else
#error does not have the constexpr builtin
#endif
// metabuiltin-error@-4 {{has the constexpr builtin}}
// nometabuiltin-error@-3 {{does not have the constexpr builtin}}
