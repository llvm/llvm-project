// RUN: %clangxx_msan %s -O0 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && MSAN_OPTIONS=poison_in_dtor=1 %run %t >%t.out 2>&1
// RUN: %clangxx_msan %s -O1 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && MSAN_OPTIONS=poison_in_dtor=1 %run %t >%t.out 2>&1
// RUN: %clangxx_msan %s -O2 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && MSAN_OPTIONS=poison_in_dtor=1  %run %t >%t.out 2>&1

#include <sanitizer/msan_interface.h>
#include <assert.h>

struct Base {
  int x;
  Base() { x = 5; }
  virtual ~Base() {}
};

struct Derived : public Base {
  int y;
  Derived() { y = 10; }
  ~Derived() {}
};

int main() {
  Derived *d = new Derived();
  d->~Derived();

  // Verify that local pointer is unpoisoned, and that the object's
  // members are.
  assert(__msan_test_shadow(&d, sizeof(d)) == -1);
  assert(__msan_test_shadow(&d->x, sizeof(d->x)) != -1);
  assert(__msan_test_shadow(&d->y, sizeof(d->y)) != -1);

  Base *b = new Derived();
  b->~Base();

  // Verify that local pointer is unpoisoned, and that the object's
  // members are.
  assert(__msan_test_shadow(&b, sizeof(b)) == -1);
  assert(__msan_test_shadow(&b->x, sizeof(b->x)) != -1);

  return 0;
}
