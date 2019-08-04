// RUN: %clangxx_msan %s -O0 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && MSAN_OPTIONS=poison_in_dtor=1 %run %t

// RUN: %clangxx_msan %s -O1 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && MSAN_OPTIONS=poison_in_dtor=1 %run %t

// RUN: %clangxx_msan %s -O2 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && MSAN_OPTIONS=poison_in_dtor=1 %run %t

#include <sanitizer/msan_interface.h>
#include <assert.h>

// TODO: remove empty dtors when msan use-after-dtor poisons
// for trivial classes with undeclared dtors

// 24 bytes total
struct Packed {
  // Packed into 4 bytes
  unsigned int a : 1;
  unsigned int b : 1;
  // Force alignment to next 4 bytes
  unsigned int   : 0;
  unsigned int c : 1;
  // Force alignment, 8 more bytes
  double d = 5.0;
  // 4 bytes
  unsigned int e : 1;
  ~Packed() {}
};

// 1 byte total
struct Empty {
  unsigned int : 0;
  ~Empty() {}
};

// 4 byte total
struct Simple {
  unsigned int a : 1;
  ~Simple() {}
};

struct Anon {
  unsigned int a : 1;
  unsigned int b : 2;
  unsigned int   : 0;
  unsigned int c : 1;
  ~Anon() {}
};

int main() {
  Packed *p = new Packed();
  p->~Packed();
  for (int i = 0; i < 4; i++)
    assert(__msan_test_shadow(((char*)p) + i, sizeof(char)) != -1);
  assert(__msan_test_shadow(&p->d, sizeof(double)) != -1);
  assert(__msan_test_shadow(((char*)(&p->d)) + sizeof(double), sizeof(char)) !=
         -1);

  Empty *e = new Empty();
  e->~Empty();
  assert(__msan_test_shadow(e, sizeof(*e)) != -1);

  Simple *s = new Simple();
  s->~Simple();
  assert(__msan_test_shadow(s, sizeof(*s)) != -1);

  Anon *a = new Anon();
  a->~Anon();
  assert(__msan_test_shadow(a, sizeof(*a)) != -1);

  return 0;
}
