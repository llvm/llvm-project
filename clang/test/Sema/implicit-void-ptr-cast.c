// RUN: %clang_cc1 -fsyntax-only -std=c23 -verify=c -Wimplicit-void-ptr-cast %s
// RUN: %clang_cc1 -fsyntax-only -std=c23 -verify=c -Wc++-compat %s
// RUN: %clang_cc1 -fsyntax-only -verify=cxx -x c++ %s
// RUN: %clang_cc1 -fsyntax-only -std=c23 -verify=good %s
// RUN: %clang_cc1 -fsyntax-only -std=c23 -verify=good -Wc++-compat -Wno-implicit-void-ptr-cast %s
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

void more(void) {
  __attribute__((address_space(0))) char *b1 = (void *)0; // c-warning {{implicit conversion when initializing '__attribute__((address_space(0))) char *' with an expression of type 'void *' is not permitted in C++}} \
                                                             cxx-error {{cannot initialize a variable of type '__attribute__((address_space(0))) char *' with an rvalue of type 'void *'}}
  __attribute__((address_space(0))) void *b2 = (void *)0; // c-warning {{implicit conversion when initializing '__attribute__((address_space(0))) void *' with an expression of type 'void *' is not permitted in C++}} \
                                                             cxx-error {{cannot initialize a variable of type '__attribute__((address_space(0))) void *' with an rvalue of type 'void *'}}
  char *b3 = (void *)0; // c-warning {{implicit conversion when initializing 'char *' with an expression of type 'void *' is not permitted in C++}} \
                           cxx-error {{cannot initialize a variable of type 'char *' with an rvalue of type 'void *'}}

  b1 = (void*)0; // c-warning {{implicit conversion when assigning to '__attribute__((address_space(0))) char *' from type 'void *' is not permitted in C++}} \
                    cxx-error {{assigning 'void *' to '__attribute__((address_space(0))) char *' changes address space of pointer}}

  b2 = (void*)0; // c-warning {{implicit conversion when assigning to '__attribute__((address_space(0))) void *' from type 'void *' is not permitted in C++}} \
                    cxx-error {{assigning 'void *' to '__attribute__((address_space(0))) void *' changes address space of pointer}}
  b2 = (__attribute__((address_space(0))) void *)0;
  b2 = nullptr;
  b2 = 0;

  b3 = (void*)0; // c-warning {{implicit conversion when assigning to 'char *' from type 'void *' is not permitted in C++}} \
                    cxx-error {{assigning to 'char *' from incompatible type 'void *'}}
  b3 = (char *)0;
  b3 = nullptr;
  b3 = 0;

  // Note that we explicitly silence the diagnostic if the RHS is from a macro
  // expansion. This allows for things like NULL expanding to different token
  // sequences depending on language mode, but applies to any macro that
  // expands to a valid null pointer constant.
#if defined(__cplusplus)
  #define NULL 0
#else
  #define NULL ((void *)0)
#endif
  #define SOMETHING_NOT_SPELLED_NULL nullptr
  #define SOMETHING_THAT_IS_NOT_NULL (void *)12

  char *ptr1 = NULL; // Ok
  char *ptr2 = SOMETHING_NOT_SPELLED_NULL; // Ok
  char *ptr3 = SOMETHING_THAT_IS_NOT_NULL; // c-warning {{implicit conversion when initializing 'char *' with an expression of type 'void *' is not permitted in C++}} \
                                              cxx-error {{cannot initialize a variable of type 'char *' with an rvalue of type 'void *'}}

  ptr1 = NULL; // Ok
  ptr2 = SOMETHING_NOT_SPELLED_NULL; // Ok
  ptr3 = SOMETHING_THAT_IS_NOT_NULL; // c-warning {{implicit conversion when assigning to 'char *' from type 'void *' is not permitted in C++}} \
                                        cxx-error {{assigning to 'char *' from incompatible type 'void *'}}
}
