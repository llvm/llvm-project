// RUN: %clang_cc1 -fsyntax-only -verify=c -Wimplicit-void-ptr-cast %s
// RUN: %clang_cc1 -fsyntax-only -verify=c -Wc++-compat %s
// RUN: %clang_cc1 -fsyntax-only -verify=cxx -x c++ %s
// RUN: %clang_cc1 -fsyntax-only -verify=good %s
// RUN: %clang_cc1 -fsyntax-only -verify=good -Wc++-compat -Wno-implicit-void-ptr-cast %s
// good-no-diagnostics

typedef __typeof__(sizeof(int)) size_t;
extern void *malloc(size_t);

void func(int *); // #func-param

void test(void) {
  int *x = malloc(sizeof(char)); // c-warning {{implicit conversion when initializing 'int *' with an expression of type 'void *' is not permitted in C++}} \
                                    cxx-error {{cannot initialize a variable of type 'int *' with an rvalue of type 'void *'}}
  x = malloc(sizeof(char));      // c-warning {{implicit conversion when assigning to 'int *' from type 'void *' is not permitted in C++}} \
                                    cxx-error {{assigning to 'int *' from incompatible type 'void *'}}
  func(malloc(sizeof(char)));    // c-warning {{implicit conversion when passing 'void *' to parameter of type 'int *' is not permitted in C++}} \
                                    c-note@#func-param {{passing argument to parameter here}} \
                                    cxx-error {{no matching function for call to 'func'}} \
                                    cxx-note@#func-param {{candidate function not viable: cannot convert argument of incomplete type 'void *' to 'int *' for 1st argument}}
  x = (int *)malloc(sizeof(char));

  void *vp = 0;
  x = vp; // c-warning {{implicit conversion when assigning to 'int *' from type 'void *' is not permitted in C++}} \
             cxx-error {{assigning to 'int *' from incompatible type 'void *'}}
  vp = vp;

  x = (void *)malloc(sizeof(char)); // c-warning {{implicit conversion when assigning to 'int *' from type 'void *' is not permitted in C++}} \
                                       cxx-error {{assigning to 'int *' from incompatible type 'void *'}}
  const int *y = vp;                // c-warning {{implicit conversion when initializing 'const int *' with an expression of type 'void *' is not permitted in C++}} \
                                       cxx-error {{cannot initialize a variable of type 'const int *' with an lvalue of type 'void *'}}
}

int *other_func(void *ptr) {
  return ptr; // c-warning {{implicit conversion when returning 'void *' from a function with result type 'int *' is not permitted in C++}} \
                 cxx-error {{cannot initialize return object of type 'int *' with an lvalue of type 'void *'}}
}
