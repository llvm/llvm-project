// RUN: %clang_cc1 -x c -triple x86_64-pc-linux-gnu -dwarf-version=4 -fsyntax-only -verify %s

const void *invalid1(const int *arg) {
  return __builtin_preserve_access_index(&arg[1], 1); // expected-error {{too many arguments to function call, expected 1, have 2}}
}

void *invalid2(const int *arg) {
  return __builtin_preserve_access_index(&arg[1]); // expected-warning {{returning 'const void *' from a function with result type 'void *' discards qualifiers}}
}

const void *invalid3(const int *arg) {
  return __builtin_preserve_access_index(1); // expected-warning {{incompatible integer to pointer conversion passing 'int' to parameter of type 'const void *'}}
}
