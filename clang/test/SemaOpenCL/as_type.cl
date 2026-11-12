// RUN: %clang_cc1 %s -DBITS=32 -triple spir-unknown-unknown -finclude-default-header -fdeclare-opencl-builtins -o - -verify -fsyntax-only
// RUN: %clang_cc1 %s -DBITS=64 -triple spir64-unknown-unknown -finclude-default-header -fdeclare-opencl-builtins -o - -verify -fsyntax-only

char3 f1(char16 x) {
  return  __builtin_astype(x, char3); // expected-error{{invalid reinterpretation: sizes of 'char3' (vector of 3 'char' values) and '__private char16' (vector of 16 'char' values) must match}}
}

char16 f3(int x) {
  return __builtin_astype(x, char16); // expected-error{{invalid reinterpretation: sizes of 'char16' (vector of 16 'char' values) and '__private int' must match}}
}

void foo() {
    char src = 1;
    int dst = as_int(src); // expected-error{{invalid reinterpretation: sizes of 'int' and '__private char' must match}}
}

void target_dependent(int i, long l) {
  size_t size1 = as_size_t(i);
#if BITS == 64
  // expected-error@-2{{sizes of 'size_t' (aka 'unsigned long') and '__private int' must match}}
#endif

  size_t size2 = as_size_t(l);
#if BITS == 32
  // expected-error@-2{{sizes of 'size_t' (aka 'unsigned int') and '__private long' must match}}
#endif
}
