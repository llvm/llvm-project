// RUN: %clang_analyze_cc1 -std=c++14 -analyzer-checker=optin.osx.OSObjectCStyleCast %s -verify
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
  OSArray *a = (OSArray *) obj; // expected-warning{{C-style cast of an OSObject is prone to type confusion attacks; use 'OSRequiredCast' if the object is definitely of type 'OSArray', or 'OSDynamicCast' followed by a null check if unsure}}
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

__SIZE_TYPE__ no_warn_on_primitive_conversion(OSArray *arr) {
  return (__SIZE_TYPE__) arr;
}

unsigned no_warn_on_other_type_cast(A *a) {
  B *b = (B *) a;
  return b->getCount();
}

unsigned no_warn_alloc_class_with_name() {
  OSArray *a = (OSArray *)OSMetaClass::allocClassWithName("OSArray"); // no warning
  return a->getCount();
}

unsigned warn_alloc_class_with_name() {
  OSArray *a = (OSArray *)OSMetaClass::allocClassWithName("OSObject"); // expected-warning{{C-style cast of an OSObject is prone to type confusion attacks; use 'OSRequiredCast' if the object is definitely of type 'OSArray', or 'OSDynamicCast' followed by a null check if unsure}}
  return a->getCount();
}
