// RUN: %clang_cc1 %s -emit-llvm -triple spir-unknown-unknown -finclude-default-header -o - -verify -fsyntax-only

char3 f1(char16 x) {
  return  __builtin_astype(x, char3); // expected-error{{invalid reinterpretation: sizes of 'char3' (vector of 3 'char' values) and 'char16' (vector of 16 'char' values) must match}}
}

char16 f3(int x) {
  return __builtin_astype(x, char16); // expected-error{{invalid reinterpretation: sizes of 'char16' (vector of 16 'char' values) and 'int' must match}}
}

void foo() {
    char src = 1;
    int dst = as_int(src); // expected-error{{invalid reinterpretation: sizes of 'int' and 'char' must match}}
}
