// RUN: %clangxx_msan %s -O0 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && MSAN_OPTIONS=poison_in_dtor=1 %run %t >%t.out 2>&1

// RUN: %clangxx_msan %s -O1 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && MSAN_OPTIONS=poison_in_dtor=1 %run %t >%t.out 2>&1

// RUN: %clangxx_msan %s -O2 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && MSAN_OPTIONS=poison_in_dtor=1 %run %t >%t.out 2>&1

#include <sanitizer/msan_interface.h>
#include <assert.h>

class Base {
 public:
  int *x_ptr;
  Base(int *y_ptr) {
    // store value of subclass member
    x_ptr = y_ptr;
  }
  virtual ~Base();
};

class Derived : public Base {
 public:
  int y;
  Derived():Base(&y) {
    y = 10;
  }
  ~Derived();
};

Base::~Base() {
  // ok access its own member
  assert(__msan_test_shadow(&this->x_ptr, sizeof(this->x_ptr)) == -1);
  // bad access subclass member
  assert(__msan_test_shadow(this->x_ptr, sizeof(*this->x_ptr)) != -1);
}

Derived::~Derived() {
  // ok to access its own members
  assert(__msan_test_shadow(&this->y, sizeof(this->y)) == -1);
  // ok access base class members
  assert(__msan_test_shadow(&this->x_ptr, sizeof(this->x_ptr)) == -1);
}

int main() {
  Derived *d = new Derived();
  assert(__msan_test_shadow(&d->x_ptr, sizeof(d->x_ptr)) == -1);
  d->~Derived();
  assert(__msan_test_shadow(&d->x_ptr, sizeof(d->x_ptr)) != -1);
  return 0;
}
