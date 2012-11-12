// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -fblocks -Wnull-arithmetic -verify -Wno-string-plus-int %s
#include <stddef.h>

void f() {
  int a;
  bool b;
  void (^c)();
  class X;
  void (X::*d) ();
  extern void e();
  int f[2];
  const void *v;

  a = 0 ? NULL + a : a + NULL; // expected-warning 2{{use of NULL in arithmetic operation}}
  a = 0 ? NULL - a : a - NULL; // expected-warning 2{{use of NULL in arithmetic operation}}
  a = 0 ? NULL / a : a / NULL; // expected-warning 2{{use of NULL in arithmetic operation}} \
                               // expected-warning {{division by zero is undefined}}
  a = 0 ? NULL * a : a * NULL; // expected-warning 2{{use of NULL in arithmetic operation}}
  a = 0 ? NULL >> a : a >> NULL; // expected-warning 2{{use of NULL in arithmetic operation}}
  a = 0 ? NULL << a : a << NULL; // expected-warning 2{{use of NULL in arithmetic operation}}
  a = 0 ? NULL % a : a % NULL; // expected-warning 2{{use of NULL in arithmetic operation}} \
                                  expected-warning {{remainder by zero is undefined}}
  a = 0 ? NULL & a : a & NULL; // expected-warning 2{{use of NULL in arithmetic operation}}
  a = 0 ? NULL | a : a | NULL; // expected-warning 2{{use of NULL in arithmetic operation}}
  a = 0 ? NULL ^ a : a ^ NULL; // expected-warning 2{{use of NULL in arithmetic operation}}

  // Check for warnings or errors when doing arithmetic on pointers and other
  // types.
  v = 0 ? NULL + &a : &a + NULL; // expected-warning 2{{use of NULL in arithmetic operation}}
  v = 0 ? NULL + c : c + NULL; // \
    expected-error {{invalid operands to binary expression ('long' and 'void (^)()')}} \
    expected-error {{invalid operands to binary expression ('void (^)()' and 'long')}}
  v = 0 ? NULL + d : d + NULL; // \
    expected-error {{invalid operands to binary expression ('long' and 'void (X::*)()')}} \
    expected-error {{invalid operands to binary expression ('void (X::*)()' and 'long')}}
  v = 0 ? NULL + e : e + NULL; // expected-error 2{{arithmetic on a pointer to the function type 'void ()'}}
  v = 0 ? NULL + f : f + NULL; // expected-warning 2{{use of NULL in arithmetic operation}}
  v = 0 ? NULL + "f" : "f" + NULL; // expected-warning 2{{use of NULL in arithmetic operation}}

  // Using two NULLs should only give one error instead of two.
  a = NULL + NULL; // expected-warning{{use of NULL in arithmetic operation}}
  a = NULL - NULL; // expected-warning{{use of NULL in arithmetic operation}}
  a = NULL / NULL; // expected-warning{{use of NULL in arithmetic operation}} \
                   // expected-warning{{division by zero is undefined}}
  a = NULL * NULL; // expected-warning{{use of NULL in arithmetic operation}}
  a = NULL >> NULL; // expected-warning{{use of NULL in arithmetic operation}}
  a = NULL << NULL; // expected-warning{{use of NULL in arithmetic operation}}
  a = NULL % NULL; // expected-warning{{use of NULL in arithmetic operation}} \
                   // expected-warning{{remainder by zero is undefined}}
  a = NULL & NULL; // expected-warning{{use of NULL in arithmetic operation}}
  a = NULL | NULL; // expected-warning{{use of NULL in arithmetic operation}}
  a = NULL ^ NULL; // expected-warning{{use of NULL in arithmetic operation}}

  a += NULL; // expected-warning{{use of NULL in arithmetic operation}}
  a -= NULL; // expected-warning{{use of NULL in arithmetic operation}}
  a /= NULL; // expected-warning{{use of NULL in arithmetic operation}} \
             // expected-warning{{division by zero is undefined}}
  a *= NULL; // expected-warning{{use of NULL in arithmetic operation}}
  a >>= NULL; // expected-warning{{use of NULL in arithmetic operation}}
  a <<= NULL; // expected-warning{{use of NULL in arithmetic operation}}
  a %= NULL; // expected-warning{{use of NULL in arithmetic operation}} \
             // expected-warning{{remainder by zero is undefined}}
  a &= NULL; // expected-warning{{use of NULL in arithmetic operation}}
  a |= NULL; // expected-warning{{use of NULL in arithmetic operation}}
  a ^= NULL; // expected-warning{{use of NULL in arithmetic operation}}

  b = a < NULL || a > NULL; // expected-warning 2{{comparison between NULL and non-pointer ('int' and NULL)}}
  b = NULL < a || NULL > a; // expected-warning 2{{comparison between NULL and non-pointer (NULL and 'int')}}
  b = a <= NULL || a >= NULL; // expected-warning 2{{comparison between NULL and non-pointer ('int' and NULL)}}
  b = NULL <= a || NULL >= a; // expected-warning 2{{comparison between NULL and non-pointer (NULL and 'int')}}
  b = a == NULL || a != NULL; // expected-warning 2{{comparison between NULL and non-pointer ('int' and NULL)}}
  b = NULL == a || NULL != a; // expected-warning 2{{comparison between NULL and non-pointer (NULL and 'int')}}

  b = &a < NULL || NULL < &a || &a > NULL || NULL > &a;
  b = &a <= NULL || NULL <= &a || &a >= NULL || NULL >= &a;
  b = &a == NULL || NULL == &a || &a != NULL || NULL != &a;

  b = 0 == a;
  b = 0 == &a;

  b = NULL < NULL || NULL > NULL;
  b = NULL <= NULL || NULL >= NULL;
  b = NULL == NULL || NULL != NULL;

  b = ((NULL)) != a;  // expected-warning{{comparison between NULL and non-pointer (NULL and 'int')}}

  // Check that even non-standard pointers don't warn.
  b = c == NULL || NULL == c || c != NULL || NULL != c;
  b = d == NULL || NULL == d || d != NULL || NULL != d;
  b = e == NULL || NULL == e || e != NULL || NULL != e;
  b = f == NULL || NULL == f || f != NULL || NULL != f;
  b = "f" == NULL || NULL == "f" || "f" != NULL || NULL != "f";

  return NULL; // expected-error{{void function 'f' should not return a value}}
}
