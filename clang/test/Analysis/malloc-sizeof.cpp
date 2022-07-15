// RUN: %clang_analyze_cc1 -analyzer-checker=unix.MallocSizeof -verify %s

#include <stddef.h>

void *malloc(size_t size);
void *calloc(size_t nmemb, size_t size);
void *realloc(void *ptr, size_t size);
void free(void *ptr);

struct A {};
struct B {};

void foo(unsigned int unsignedInt, unsigned int readSize) {
  // Verify the checker is working as expected.
  A* a = static_cast<A*>(malloc(sizeof(int))); // expected-warning {{Result of 'malloc' is converted to a pointer of type 'struct A', which is incompatible with sizeof operand type 'int'}}
  free(a);
}

void bar() {
  A *x = static_cast<A*>(calloc(10, sizeof(void*))); // expected-warning {{Result of 'calloc' is converted to a pointer of type 'struct A', which is incompatible with sizeof operand type 'void *'}}
  // sizeof(void*) is compatible with any pointer.
  A **y = static_cast<A**>(calloc(10, sizeof(void*))); // no-warning
  free(x);
  free(y);
}

