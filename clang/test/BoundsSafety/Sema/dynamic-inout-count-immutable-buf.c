
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

void foo_c(int *__counted_by(*len) buf, int *len) {
  buf = buf;
  *len = *len - 1;
}

void bar_c(int *__counted_by(*len) buf, int *len) {
  // expected-error@+1{{parameter 'buf' with '__counted_by' attribute depending on an indirect count is implicitly read-only}}
  buf = buf + 1;
  *len = *len - 1;
}

void baz_c(int *__counted_by(*len) buf, int *len) {
  // expected-error@+1{{parameter 'buf' with '__counted_by' attribute depending on an indirect count is implicitly read-only}}
  buf++;
  *len = *len - 1;
}

void foo_s(int *__sized_by(*size) buf, int *size) {
  buf = buf;
  *size = *size - 4;
}

void bar_s(int *__sized_by(*size) buf, int *size) {
  // expected-error@+1{{parameter 'buf' with '__sized_by' attribute depending on an indirect count is implicitly read-only}}
  buf = buf + 1;
  *size = *size - 4;
}

void baz_s(int *__sized_by(*size) buf, int *size) {
  // expected-error@+1{{parameter 'buf' with '__sized_by' attribute depending on an indirect count is implicitly read-only}}
  buf++;
  *size = *size - 4;
}

void foo_cn(int *__counted_by_or_null(*len) buf, int *len) {
  buf = buf;
  *len = *len - 1;
}

void bar_cn(int *__counted_by_or_null(*len) buf, int *len) {
  // expected-error@+1{{parameter 'buf' with '__counted_by_or_null' attribute depending on an indirect count is implicitly read-only}}
  buf = buf + 1;
  *len = *len - 1;
}

void baz_cn(int *__counted_by_or_null(*len) buf, int *len) {
  // expected-error@+1{{parameter 'buf' with '__counted_by_or_null' attribute depending on an indirect count is implicitly read-only}}
  buf++;
  *len = *len - 1;
}

void foo_sn(int *__sized_by_or_null(*size) buf, int *size) {
  buf = buf;
  *size = *size - 4;
}

void bar_sn(int *__sized_by_or_null(*size) buf, int *size) {
  // expected-error@+1{{parameter 'buf' with '__sized_by_or_null' attribute depending on an indirect count is implicitly read-only}}
  buf = buf + 1;
  *size = *size - 4;
}

void baz_sn(int *__sized_by_or_null(*size) buf, int *size) {
  // expected-error@+1{{parameter 'buf' with '__sized_by_or_null' attribute depending on an indirect count is implicitly read-only}}
  buf++;
  *size = *size - 4;
}
