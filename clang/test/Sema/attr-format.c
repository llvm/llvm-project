//RUN: %clang_cc1 -fsyntax-only -verify %s

#include <stdarg.h>

void a(const char *a, ...) __attribute__((format(printf, 1, 2)));    // no-error
void b(const char *a, ...) __attribute__((format(printf, 1, 1)));    // expected-error {{'format' attribute parameter 3 is out of bounds}}
void c(const char *a, ...) __attribute__((format(printf, 0, 2)));    // expected-error {{'format' attribute parameter 2 is out of bounds}}
void d(const char *a, int c) __attribute__((format(printf, 1, 2)));  // expected-warning {{GCC requires a function with the 'format' attribute to be variadic}}
void e(char *str, int c, ...) __attribute__((format(printf, 2, 3))); // expected-error {{format argument not a string type}}
void f(int a, const char *b, ...) __attribute__((format(printf, 2, 1))); // expected-error {{'format' attribute parameter 3 is out of bounds}}
void g(int a, const char *b, ...) __attribute__((format(printf, 2, 2))); // expected-error {{'format' attribute parameter 3 is out of bounds}}
void h(int a, const char *b, ...) __attribute__((format(printf, 2, 3))); // no-error
void i(const char *a, int b, ...) __attribute__((format(printf, 1, 2))); // expected-error {{'format' attribute parameter 3 is out of bounds}}

typedef const char *xpto;
void j(xpto c, va_list list) __attribute__((format(printf, 1, 0))); // no-error
void k(xpto c) __attribute__((format(printf, 1, 0)));               // no-error

void y(char *str) __attribute__((format(strftime, 1, 0)));             // no-error
void z(char *str, int c, ...) __attribute__((format(strftime, 1, 2))); // expected-error {{strftime format attribute requires 3rd parameter to be 0}}

int (*f_ptr)(char*,...) __attribute__((format(printf, 1,2))); // no-error
int (*f2_ptr)(double,...) __attribute__((format(printf, 1, 2))); // expected-error {{format argument not a string type}}

struct _mystruct {
  int (*printf)(const char *format, ...) __attribute__((__format__(printf, 1, 2))); // no-error
  int (*printf2)(double format, ...) __attribute__((__format__(printf, 1, 2))); // expected-error {{format argument not a string type}}
};

typedef int (*f3_ptr)(char*,...) __attribute__((format(printf,1,0))); // no-error

// <rdar://problem/6623513>
int rdar6623513(void *, const char*, const char*, ...)
  __attribute__ ((format (printf, 3, 0)));

int rdar6623513_aux(int len, const char* s) {
  rdar6623513(0, "hello", "%.*s", len, s);
}

// same as format(printf(...))...
void a2(const char *a, ...) __attribute__((format(printf0, 1, 2)));    // no-error
void b2(const char *a, ...) __attribute__((format(printf0, 1, 1)));    // expected-error {{'format' attribute parameter 3 is out of bounds}}
void c2(const char *a, ...) __attribute__((format(printf0, 0, 2)));    // expected-error {{'format' attribute parameter 2 is out of bounds}}
void d2(const char *a, int c) __attribute__((format(printf0, 1, 2)));  // expected-warning {{GCC requires a function with the 'format' attribute to be variadic}}
void e2(char *str, int c, ...) __attribute__((format(printf0, 2, 3))); // expected-error {{format argument not a string type}}

// FreeBSD usage
#define __printf0like(fmt, va) __attribute__((__format__(__printf0__, fmt, va)))
void null(int i, const char *a, ...) __printf0like(2, 0); // no-error
void null(int i, const char *a, ...) {                    // expected-note{{passing argument to parameter 'a' here}}
  if (a)
    (void)0/* vprintf(...) would go here */;
}

void callnull(void){
  null(0,        0); // no error
  null(0, (char*)0); // no error
  null(0, (void*)0); // no error
  null(0,  (int*)0); // expected-warning {{incompatible pointer types}}
}

// FreeBSD kernel extensions
void a3(const char *a, ...) __attribute__((format(freebsd_kprintf, 1, 2)));    // no-error
void b3(const char *a, ...) __attribute__((format(freebsd_kprintf, 1, 1)));    // expected-error {{'format' attribute parameter 3 is out of bounds}}
void c3(const char *a, ...) __attribute__((format(freebsd_kprintf, 0, 2)));    // expected-error {{'format' attribute parameter 2 is out of bounds}}
void d3(const char *a, int c) __attribute__((format(freebsd_kprintf, 1, 2)));  // expected-warning {{GCC requires a function with the 'format' attribute to be variadic}}
void e3(char *str, int c, ...) __attribute__((format(freebsd_kprintf, 2, 3))); // expected-error {{format argument not a string type}}

// PR4470
int xx_vprintf(const char *, va_list);

const char *foo(const char *format) __attribute__((format_arg(1)));

void __attribute__((format(printf, 1, 0)))
foo2(const char *fmt, va_list va) {
  xx_vprintf(foo(fmt), va);
}

// PR6542
extern void gcc_format(const char *, ...)
    __attribute__((__format__(__gcc_diag__, 1, 2)));
extern void gcc_cformat(const char *, ...)
    __attribute__((__format__(__gcc_cdiag__, 1, 2)));
extern void gcc_cxxformat(const char *, ...)
    __attribute__((__format__(__gcc_cxxdiag__, 1, 2)));
extern void gcc_tformat(const char *, ...)
    __attribute__((__format__(__gcc_tdiag__, 1, 2)));

const char *foo3(const char *format) __attribute__((format_arg("foo"))); // expected-error{{'format_arg' attribute requires parameter 1 to be an integer constant}}

void call_nonvariadic(void) {
  d3("%i", 123);
  d3("%d", 123);
  d3("%s", 123); // expected-warning{{format specifies type 'char *' but the argument has type 'int'}}
}

__attribute__((format(printf, 1, 2))) void forward_fixed(const char *fmt, int i) { // expected-warning{{GCC requires a function with the 'format' attribute to be variadic}}
  forward_fixed(fmt, i);
  a(fmt, i);
}

__attribute__((format(printf, 1, 2))) void forward_fixed_2(const char *fmt, int i, int j) { // expected-warning{{GCC requires a function with the 'format' attribute to be variadic}}
  forward_fixed_2(fmt, i, j);
  a(fmt, i);
}

