// RUN: %clang_analyze_cc1 -analyzer-checker=optin.osx.OSObjectCStyleCast %s -verify
#include "os_object_base.h"

struct OSArray : public OSObject {
  unsigned getCount();
};

struct A {
  int x;
};
struct B : public A {
  unsigned getCount();
};

unsigned warn_on_explicit_downcast(OSObject * obj) {
  OSArray *a = (OSArray *) obj; // expected-warning{{C-style cast of OSObject. Use OSDynamicCast instead}}
  return a->getCount();
}

void no_warn_on_upcast(OSArray *arr) {
  OSObject *obj = (OSObject *) arr;
  obj->retain();
  obj->release();
}

unsigned no_warn_on_dynamic_cast(OSObject *obj) {
  OSArray *a = OSDynamicCast(OSArray, obj);
  return a->getCount();
}

unsigned long no_warn_on_primitive_conversion(OSArray *arr) {
  return (unsigned long) arr;
}

unsigned no_warn_on_other_type_cast(A *a) {
  B *b = (B *) a;
  return b->getCount();
}

