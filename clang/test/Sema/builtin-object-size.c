// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-darwin9 -verify %s

int a[10];

int f0() {
  return __builtin_object_size(&a); // expected-error {{too few arguments to function}}
}
int f1() {
  return (__builtin_object_size(&a, 0) + 
          __builtin_object_size(&a, 1) + 
          __builtin_object_size(&a, 2) + 
          __builtin_object_size(&a, 3));
}
int f2() {
  return __builtin_object_size(&a, -1); // expected-error {{argument should be a value from 0 to 3}}
}
int f3() {
  return __builtin_object_size(&a, 4); // expected-error {{argument should be a value from 0 to 3}}
}


// rdar://6252231 - cannot call vsnprintf with va_list on x86_64
void f4(const char *fmt, ...) {
 __builtin_va_list args;
 __builtin___vsnprintf_chk (0, 42, 0, 11, fmt, args);
}

