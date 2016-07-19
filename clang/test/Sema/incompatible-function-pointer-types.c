// RUN: %clang_cc1 -fsyntax-only %s -Wincompatible-pointer-types -verify
// RUN: %clang_cc1 -fsyntax-only %s -Wincompatible-function-pointer-types -verify

// This test ensures that the subgroup of -Wincompatible-pointer-types warnings
// that concern function pointers can be promoted (or not promoted) to an error
// *separately* from the other -Wincompatible-pointer-type warnings.
typedef int (*MyFnTyA)(int *, char *);

int bar(char *a, int *b) { return 0; }
int foo(MyFnTyA x) { return 0; } // expected-note {{passing argument to parameter 'x' here}}

void baz() {
  foo(&bar); // expected-warning {{incompatible function pointer types passing 'int (*)(char *, int *)' to parameter of type 'MyFnTyA' (aka 'int (*)(int *, char *)')}}
}
