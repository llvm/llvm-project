// RUN: %clang_cc1 %s -verify -pedantic
typedef int unary_int_func(int arg);
unary_int_func *func;

unary_int_func *set_func(void *p) {
 func = p; // expected-warning {{converts between void pointer and function pointer}}
 p = func; // expected-warning {{converts between void pointer and function pointer}}

 return p; // expected-warning {{converts between void pointer and function pointer}}
}

