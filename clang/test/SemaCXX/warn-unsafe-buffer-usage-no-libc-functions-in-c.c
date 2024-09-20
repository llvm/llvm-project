// RUN: %clang_cc1 -Wunsafe-buffer-usage %s -verify %s

void* memcpy(void *dst,const void *src, unsigned long size);

void f(int *p, int *q) {

  memcpy(p, q, 10); // no libc warn in C
  ++p[5];           // expected-warning{{unsafe buffer access}}
}
