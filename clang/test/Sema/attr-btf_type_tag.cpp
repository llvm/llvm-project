// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify=c -x c %s

// c-no-diagnostics

// Ensure that we diagnose the attribute as ignored in C++ but not in C.
#ifdef __cplusplus
static_assert(__builtin_is_implicit_lifetime(int __attribute__((btf_type_tag("user"))) *)); // expected-warning {{'btf_type_tag' attribute ignored}}
#endif
int __attribute__((btf_type_tag("user"))) *ptr; // expected-warning {{'btf_type_tag' attribute ignored}}

