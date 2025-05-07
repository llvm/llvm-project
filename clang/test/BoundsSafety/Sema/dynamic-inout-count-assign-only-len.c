
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

void in_buf_inout_len(int *__counted_by(*len) buf, int *len) {
  *len = *len - 1;
bb:
  (*len)--;
}

// Make sure that for other cases we cannot assign only len.

void inout_buf_inout_len(int *__counted_by(*len) * out_buf, int *len) {
  // expected-error@+1{{assignment to '*len' requires corresponding assignment to 'int *__single __counted_by(*len)' (aka 'int *__single') '*out_buf'; add self assignment '*out_buf = *out_buf' if the value has not changed}}
  *len = *len - 1;
bb:
  // expected-error@+1{{assignment to '*len' requires corresponding assignment to 'int *__single __counted_by(*len)' (aka 'int *__single') '*out_buf'; add self assignment '*out_buf = *out_buf' if the value has not changed}}
  (*len)--;
}

void in_buf_inout_buf_inout_len(int *__counted_by(*len) buf, int *__counted_by(*len) * out_buf, int *len) {
  // expected-error@+1{{assignment to '*len' requires corresponding assignment to 'int *__single __counted_by(*len)' (aka 'int *__single') '*out_buf'; add self assignment '*out_buf = *out_buf' if the value has not changed}}
  *len = *len - 1;
bb:
  // expected-error@+1{{assignment to '*len' requires corresponding assignment to 'int *__single __counted_by(*len)' (aka 'int *__single') '*out_buf'; add self assignment '*out_buf = *out_buf' if the value has not changed}}
  (*len)--;
}

void inout_buf_in_buf_inout_len(int *__counted_by(*len) * out_buf, int *__counted_by(*len) buf, int *len) {
  // expected-error@+1{{assignment to '*len' requires corresponding assignment to 'int *__single __counted_by(*len)' (aka 'int *__single') '*out_buf'; add self assignment '*out_buf = *out_buf' if the value has not changed}}
  *len = *len - 1;
bb:
  // expected-error@+1{{assignment to '*len' requires corresponding assignment to 'int *__single __counted_by(*len)' (aka 'int *__single') '*out_buf'; add self assignment '*out_buf = *out_buf' if the value has not changed}}
  (*len)--;
}

void in_buf_in_len(int *__counted_by(len) buf, int len) {
  // expected-error@+1{{assignment to 'len' requires corresponding assignment to 'int *__single __counted_by(len)' (aka 'int *__single') 'buf'; add self assignment 'buf = buf' if the value has not changed}}
  len = len - 1;
bb:
  // expected-error@+1{{assignment to 'len' requires corresponding assignment to 'int *__single __counted_by(len)' (aka 'int *__single') 'buf'; add self assignment 'buf = buf' if the value has not changed}}
  len--;
}
