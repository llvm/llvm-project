// RUN: %clang_cc1 -fsyntax-only -verify -Wimplicit-void-ptr-cast %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wc++-compat %s
// RUN: %clang_cc1 -fsyntax-only -verify=good %s
// RUN: %clang_cc1 -fsyntax-only -verify=good -Wc++-compat -Wno-implicit-void-ptr-cast %s
// good-no-diagnostics

typedef __typeof__(sizeof(int)) size_t;
extern void *malloc(size_t);

void func(int *); // expected-note {{passing argument to parameter here}}

void test(void) {
  int *x = malloc(sizeof(char)); // expected-warning {{implicit conversion when initializing 'int *' with an expression of type 'void *' is not permitted in C++}}
  x = malloc(sizeof(char)); // expected-warning {{implicit conversion when assigning to 'int *' from type 'void *' is not permitted in C++}}
  func(malloc(sizeof(char))); // expected-warning {{implicit conversion when passing 'void *' to parameter of type 'int *' is not permitted in C++}}
  x = (int *)malloc(sizeof(char));

  void *vp = 0;
  x = vp; // expected-warning {{implicit conversion when assigning to 'int *' from type 'void *' is not permitted in C++}}
  vp = vp;

  x = (void *)malloc(sizeof(char)); // expected-warning {{implicit conversion when assigning to 'int *' from type 'void *' is not permitted in C++}}
  const int *y = vp; // expected-warning {{implicit conversion when initializing 'const int *' with an expression of type 'void *' is not permitted in C++}}
}

int *other_func(void *ptr) {
  return ptr; // expected-warning {{implicit conversion when returning 'void *' from a function with result type 'int *' is not permitted in C++}}
}
