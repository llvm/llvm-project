// RUN: %clangxx_msan %s -O0 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && %run %t >%t.out 2>&1

// RUN: %clangxx_msan %s -O1 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && %run %t >%t.out 2>&1

// RUN: %clangxx_msan %s -O2 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && %run %t >%t.out 2>&1

#include <sanitizer/msan_interface.h>
#include <assert.h>

template <class T> class Vector {
public:
  int size;
  ~Vector() {
    assert(__msan_test_shadow(&this->size, sizeof(this->size)) == -1);
  }
};

struct VirtualBase {
public:
  Vector<int> virtual_v;
  int virtual_a;
  // Pointer to subclass member
  int *intermediate_a_ptr;

  VirtualBase() {
    virtual_v.size = 1;
    virtual_a = 9;
  }
  void set_ptr(int *intermediate_a) {
    this->intermediate_a_ptr = intermediate_a;
  }
  virtual ~VirtualBase() {
    assert(__msan_test_shadow(&virtual_v, sizeof(virtual_v)) == -1);
    assert(__msan_test_shadow(&virtual_a, sizeof(virtual_a)) == -1);
    // Derived class member is poisoned
    assert(__msan_test_shadow(intermediate_a_ptr,
                              sizeof(*intermediate_a_ptr)) != -1);
  }
};

struct Intermediate : virtual public VirtualBase {
public:
  int intermediate_a;

  Intermediate() { intermediate_a = 5; }
  virtual ~Intermediate() {
    assert(__msan_test_shadow(&this->intermediate_a,
                              sizeof(this->intermediate_a)) == -1);
    // Members inherited from VirtualBase unpoisoned
    assert(__msan_test_shadow(&virtual_v, sizeof(virtual_v)) == -1);
    assert(__msan_test_shadow(&virtual_a, sizeof(virtual_a)) == -1);
    assert(__msan_test_shadow(intermediate_a_ptr,
                              sizeof(*intermediate_a_ptr)) == -1);
  }
};

struct Base {
  int base_a;
  Vector<int> base_v;
  double base_b;
  // Pointers to subclass members
  int *derived_a_ptr;
  Vector<int> *derived_v1_ptr;
  Vector<int> *derived_v2_ptr;
  double *derived_b_ptr;
  double *derived_c_ptr;

  Base(int *derived_a, Vector<int> *derived_v1, Vector<int> *derived_v2,
       double *derived_b, double *derived_c) {
    base_a = 2;
    base_v.size = 1;
    base_b = 13.2324;
    derived_a_ptr = derived_a;
    derived_v1_ptr = derived_v1;
    derived_v2_ptr = derived_v2;
    derived_b_ptr = derived_b;
    derived_c_ptr = derived_c;
  }
  virtual ~Base() {
    assert(__msan_test_shadow(&base_a, sizeof(base_a)) == -1);
    assert(__msan_test_shadow(&base_v, sizeof(base_v)) == -1);
    assert(__msan_test_shadow(&base_b, sizeof(base_b)) == -1);
    // Derived class members are poisoned
    assert(__msan_test_shadow(derived_a_ptr, sizeof(*derived_a_ptr)) != -1);
    assert(__msan_test_shadow(derived_v1_ptr, sizeof(*derived_v1_ptr)) != -1);
    assert(__msan_test_shadow(derived_v2_ptr, sizeof(*derived_v2_ptr)) != -1);
    assert(__msan_test_shadow(derived_b_ptr, sizeof(*derived_b_ptr)) != -1);
    assert(__msan_test_shadow(derived_c_ptr, sizeof(*derived_c_ptr)) != -1);
  }
};

struct Derived : public Base, public Intermediate {
  int derived_a;
  Vector<int> derived_v1;
  Vector<int> derived_v2;
  double derived_b;
  double derived_c;

  Derived()
      : Base(&derived_a, &derived_v1, &derived_v2, &derived_b, &derived_c) {
    derived_a = 5;
    derived_v1.size = 1;
    derived_v2.size = 1;
    derived_b = 7;
    derived_c = 10;
  }
  ~Derived() {
    assert(__msan_test_shadow(&derived_a, sizeof(derived_a)) == -1);
    assert(__msan_test_shadow(&derived_v1, sizeof(derived_v1)) == -1);
    assert(__msan_test_shadow(&derived_v2, sizeof(derived_v2)) == -1);
    assert(__msan_test_shadow(&derived_b, sizeof(derived_b)) == -1);
    assert(__msan_test_shadow(&derived_c, sizeof(derived_c)) == -1);
  }
};

int main() {
  Derived *d = new Derived();
  d->set_ptr(&d->intermediate_a);

  // Keep track of members of VirtualBase, since the virtual base table
  // is inaccessible after destruction
  Vector<int> *temp_virtual_v = &d->virtual_v;
  int *temp_virtual_a = &d->virtual_a;
  int **temp_intermediate_a_ptr = &d->intermediate_a_ptr;

  d->~Derived();
  assert(__msan_test_shadow(&d->derived_a, sizeof(d->derived_a)) != -1);
  assert(__msan_test_shadow(&d->derived_v1, sizeof(d->derived_v1)) != -1);
  assert(__msan_test_shadow(&d->derived_v2, sizeof(d->derived_v2)) != -1);
  assert(__msan_test_shadow(&d->derived_b, sizeof(d->derived_b)) != -1);
  assert(__msan_test_shadow(&d->derived_c, sizeof(d->derived_c)) != -1);

  // Inherited from base
  assert(__msan_test_shadow(&d->base_a, sizeof(d->base_a)) != -1);
  assert(__msan_test_shadow(&d->base_v, sizeof(d->base_v)) != -1);
  assert(__msan_test_shadow(&d->base_b, sizeof(d->base_b)) != -1);
  assert(__msan_test_shadow(&d->derived_a_ptr, sizeof(d->derived_a_ptr)) != -1);
  assert(__msan_test_shadow(&d->derived_v1_ptr, sizeof(d->derived_v1_ptr)) !=
         -1);
  assert(__msan_test_shadow(&d->derived_v2_ptr, sizeof(d->derived_v2_ptr)) !=
         -1);
  assert(__msan_test_shadow(&d->derived_b_ptr, sizeof(d->derived_b_ptr)) != -1);
  assert(__msan_test_shadow(&d->derived_c_ptr, sizeof(d->derived_c_ptr)) != -1);

  // Inherited from intermediate
  assert(__msan_test_shadow(temp_virtual_v, sizeof(*temp_virtual_v)) != -1);
  assert(__msan_test_shadow(temp_virtual_a, sizeof(*temp_virtual_a)) != -1);
  assert(__msan_test_shadow(temp_intermediate_a_ptr,
                            sizeof(*temp_intermediate_a_ptr)) != -1);

  return 0;
}
