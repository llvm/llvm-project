// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

struct B {
  int x, y, z, w;
} b;

struct __attribute__((packed)) A {
  struct B b;
} a;

typedef __typeof__(sizeof(int)) size_t;

void *memcpy(void *dest, const void *src, size_t n);
int memcmp(const void *s1, const void *s2, size_t n);
void *memmove(void *dest, const void *src, size_t n);
void *memset(void *s, int c, size_t n);

int x;

void foo(void) {
  memcpy(&a.b, &b, sizeof(b));
  memmove(&a.b, &b, sizeof(b));
  memset(&a.b, 0, sizeof(b));
  x = memcmp(&a.b, &b, sizeof(b));
}
