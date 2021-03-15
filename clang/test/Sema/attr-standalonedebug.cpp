// RUN: %clang_cc1 %s -verify -fsyntax-only
// RUN: %clang_cc1 %s -verify -fsyntax-only -x c

#ifdef __cplusplus
int a __attribute__((standalone_debug)); // expected-warning {{'standalone_debug' attribute only applies to classes}}

void __attribute__((standalone_debug)) b(); // expected-warning {{'standalone_debug' attribute only applies to classes}}

struct __attribute__((standalone_debug(1))) c {}; // expected-error {{'standalone_debug' attribute takes no arguments}}

#else
// Check that attribute only works in C++.
struct __attribute__((standalone_debug)) a {}; // expected-warning {{'standalone_debug' attribute ignored}}
#endif
