// Defines diamond multiple inheritance structure
//   A
//  / \
// B   C
//  \ /
//   Derived

// RUN: %clangxx_msan %s -O0 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && %run %t >%t.out 2>&1

// RUN: %clangxx_msan %s -O1 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && %run %t >%t.out 2>&1

// RUN: %clangxx_msan %s -O2 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && %run %t >%t.out 2>&1

#include <sanitizer/msan_interface.h>
#include <assert.h>

int *temp_x;
int *temp_y;
int *temp_z;
int *temp_w;

class A {
public:
  int x;
  A() { x = 5; }
  virtual ~A() {
    assert(__msan_test_shadow(&this->x, sizeof(this->x) == -1));
    // Memory owned by subclasses is poisoned.
    assert(__msan_test_shadow(temp_y, sizeof(*temp_y)) != -1);
    assert(__msan_test_shadow(temp_z, sizeof(*temp_z)) != -1);
    assert(__msan_test_shadow(temp_w, sizeof(*temp_w)) != -1);
  }
};

struct B : virtual public A {
public:
  int y;
  B() { y = 10; }
  virtual ~B() {
    assert(__msan_test_shadow(&this->y, sizeof(this->y)) == -1);
    // Memory accessible via vtable still reachable.
    assert(__msan_test_shadow(&this->x, sizeof(this->x)) == -1);
    // Memory in sibling and subclass is poisoned.
    assert(__msan_test_shadow(temp_z, sizeof(*temp_z)) != -1);
    assert(__msan_test_shadow(temp_w, sizeof(*temp_w)) != -1);
  }
};

struct C : virtual public A {
public:
  int z;
  C() { z = 15; }
  virtual ~C() {
    assert(__msan_test_shadow(&this->z, sizeof(this->z)) == -1);
    // Memory accessible via vtable still reachable.
    assert(__msan_test_shadow(&this->x, sizeof(this->x)) == -1);
    // Sibling class is unpoisoned.
    assert(__msan_test_shadow(temp_y, sizeof(*temp_y)) == -1);
    // Memory in subclasses is poisoned.
    assert(__msan_test_shadow(temp_w, sizeof(*temp_w)) != -1);
  }
};

class Derived : public B, public C {
public:
  int w;
  Derived() { w = 10; }
  ~Derived() {
    assert(__msan_test_shadow(&this->x, sizeof(this->x)) == -1);
    // Members accessed through the vtable are still accessible.
    assert(__msan_test_shadow(&this->y, sizeof(this->y)) == -1);
    assert(__msan_test_shadow(&this->z, sizeof(this->z)) == -1);
    assert(__msan_test_shadow(&this->w, sizeof(this->w)) == -1);
  }
};


int main() {
  Derived *d = new Derived();

  // Keep track of members inherited from virtual bases,
  // since the virtual base table is inaccessible after destruction.
  temp_x = &d->x;
  temp_y = &d->y;
  temp_z = &d->z;
  temp_w = &d->w;

  // Order of destruction: Derived, C, B, A
  d->~Derived();
  // Verify that local pointer is unpoisoned, and that the object's
  // members are.
  assert(__msan_test_shadow(&d, sizeof(d)) == -1);
  assert(__msan_test_shadow(temp_x, sizeof(*temp_x)) != -1);
  assert(__msan_test_shadow(temp_y, sizeof(*temp_y)) != -1);
  assert(__msan_test_shadow(temp_z, sizeof(*temp_z)) != -1);
  assert(__msan_test_shadow(temp_w, sizeof(*temp_w)) != -1);
  return 0;
}
