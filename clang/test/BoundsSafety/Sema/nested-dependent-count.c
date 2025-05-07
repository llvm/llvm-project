
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

struct S {
  // expected-error@+1{{'__counted_by' attribute requires an integer type argument}}
  int *__counted_by(len) *__counted_by(len) buf; // expected-error{{'__counted_by' attribute on nested pointer type is only allowed on indirect parameters}}
  int *len;
};

int fooOut(int *__counted_by(len) *out_buf, int len); // ok
int fooOutOut(int *__counted_by(*len) *out_buf, int *len); // ok
// expected-error@+1{{'__counted_by' attribute on nested pointer type is only allowed on indirect parameters}}
int bar(int *__counted_by(len) *__counted_by(len) out_buf, int len);

int baz() {
  int len;
  int *__counted_by(len) buf;

  // expected-error@+1{{passing address of 'buf' as an indirect parameter; must also pass 'len' or its address because the type of 'buf', 'int *__single __counted_by(len)' (aka 'int *__single'), refers to 'len'}}
  fooOut(&buf, len);
  fooOutOut(&buf, &len);

  int *p = &len; // expected-error{{variable referred to by '__counted_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}

  return 0;
}

int main() {
  int len;

  {
    // expected-error@+1{{'__counted_by' attribute on nested pointer type is only allowed on indirect parameters}}
    int *__counted_by(len) *nested_buf;
  }

  len = 100;
  return 0;
}
