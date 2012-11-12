//RUN: %clang_cc1 -fsyntax-only -verify %s

#include <stdarg.h>

void a(const char *a, ...) __attribute__((format(printf, 1,2))); // no-error
void b(const char *a, ...) __attribute__((format(printf, 1,1))); // expected-error {{'format' attribute parameter 3 is out of bounds}}
void c(const char *a, ...) __attribute__((format(printf, 0,2))); // expected-error {{'format' attribute parameter 2 is out of bounds}}
void d(const char *a, int c) __attribute__((format(printf, 1,2))); // expected-error {{format attribute requires variadic function}}
void e(char *str, int c, ...) __attribute__((format(printf, 2,3))); // expected-error {{format argument not a string type}}

typedef const char* xpto;
void f(xpto c, va_list list) __attribute__((format(printf, 1, 0))); // no-error
void g(xpto c) __attribute__((format(printf, 1, 0))); // no-error

void y(char *str) __attribute__((format(strftime, 1,0))); // no-error
void z(char *str, int c, ...) __attribute__((format(strftime, 1,2))); // expected-error {{strftime format attribute requires 3rd parameter to be 0}}

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
void a2(const char *a, ...)    __attribute__((format(printf0, 1,2))); // no-error
void b2(const char *a, ...)    __attribute__((format(printf0, 1,1))); // expected-error {{'format' attribute parameter 3 is out of bounds}}
void c2(const char *a, ...)    __attribute__((format(printf0, 0,2))); // expected-error {{'format' attribute parameter 2 is out of bounds}}
void d2(const char *a, int c)  __attribute__((format(printf0, 1,2))); // expected-error {{format attribute requires variadic function}}
void e2(char *str, int c, ...) __attribute__((format(printf0, 2,3))); // expected-error {{format argument not a string type}}

// FreeBSD usage
#define __printf0like(fmt,va) __attribute__((__format__(__printf0__,fmt,va)))
void null(int i, const char *a, ...) __printf0like(2,0); // no-error
void null(int i, const char *a, ...) { // expected-note{{passing argument to parameter 'a' here}}
  if (a)
    (void)0/* vprintf(...) would go here */;
}

void callnull(void){
  null(0,        0); // no error
  null(0, (char*)0); // no error
  null(0, (void*)0); // no error
  null(0,  (int*)0); // expected-warning {{incompatible pointer types}}
}



// PR4470
int xx_vprintf(const char *, va_list);

const char *foo(const char *format) __attribute__((format_arg(1)));

void __attribute__((format(printf, 1, 0)))
foo2(const char *fmt, va_list va) {
  xx_vprintf(foo(fmt), va);
}

// PR6542
extern void gcc_format (const char *, ...)
  __attribute__ ((__format__(__gcc_diag__, 1, 2)));
extern void gcc_cformat (const char *, ...)
  __attribute__ ((__format__(__gcc_cdiag__, 1, 2)));
extern void gcc_cxxformat (const char *, ...)
  __attribute__ ((__format__(__gcc_cxxdiag__, 1, 2)));
extern void gcc_tformat (const char *, ...)
  __attribute__ ((__format__(__gcc_tdiag__, 1, 2)));
