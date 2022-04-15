// RUN: %clangxx_msan %s -O0 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && %run %t >%t.out 2>&1

// RUN: %clangxx_msan %s -O1 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && %run %t >%t.out 2>&1

// RUN: %clangxx_msan %s -O2 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && %run %t >%t.out 2>&1

#include <sanitizer/msan_interface.h>
#include <assert.h>
#include <stdio.h>

template <class T>
class Vector {
public:
  int size;
  ~Vector() {
    printf("~V %p %lu\n", &size, sizeof(size));
    assert(__msan_test_shadow(&this->size, sizeof(this->size)) == -1);
  }
};

struct Derived {
  int derived_a;
  Vector<int> derived_v1;
  Vector<int> derived_v2;
  double derived_b;
  double derived_c;
  Derived() {
    derived_a = 5;
    derived_v1.size = 1;
    derived_v2.size = 1;
    derived_b = 7;
    derived_c = 10;
  }
  ~Derived() {
    printf("~D %p %p %p %lu\n", &derived_a, &derived_v1, &derived_c, sizeof(*this));
    assert(__msan_test_shadow(&derived_a, sizeof(derived_a)) == -1);
    assert(__msan_test_shadow(&derived_v1, sizeof(derived_v1)) == -1);
    assert(__msan_test_shadow(&derived_v2, sizeof(derived_v2)) == -1);
    assert(__msan_test_shadow(&derived_b, sizeof(derived_b)) == -1);
    assert(__msan_test_shadow(&derived_c, sizeof(derived_c)) == -1);
  }
};

int main() {
  Derived *d = new Derived();
  d->~Derived();

  assert(__msan_test_shadow(&d->derived_a, sizeof(d->derived_a)) != -1);
  assert(__msan_test_shadow(&d->derived_v1, sizeof(d->derived_v1)) != -1);
  assert(__msan_test_shadow(&d->derived_v2, sizeof(d->derived_v2)) != -1);
  assert(__msan_test_shadow(&d->derived_b, sizeof(d->derived_b)) != -1);
  assert(__msan_test_shadow(&d->derived_c, sizeof(d->derived_c)) != -1);

  return 0;
}
