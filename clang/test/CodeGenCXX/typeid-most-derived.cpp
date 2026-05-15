// RUN: %clang_cc1 %s -triple %itanium_abi_triple -Wno-unused-value -emit-llvm -o - -std=c++11 | FileCheck %s

namespace std {
  class type_info {};
}

struct Base {
  virtual int foo() { return 42; }
  virtual ~Base();
};

struct NonFinal : Base {};
struct Final final : Base {
    int foo() override { return 84; }
};

// Most derived
void base_by_value(Base b) { typeid(b); }
// CHECK-LABEL: define {{.*}}void @_Z13base_by_value4Base
// CHECK-NOT:   %vtable
// CHECK:       ret void

// Most derived
void final_ref(Final &f) { typeid(f); }
// CHECK-LABEL: define {{.*}}void @_Z9final_refR5Final
// CHECK-NOT:   %vtable
// CHECK:       ret void

// Most derived
void final_deref(Final *f) { typeid(*f); }
// CHECK-LABEL: define {{.*}}void @_Z11final_derefP5Final
// CHECK-NOT:   %vtable
// CHECK:       ret void

// Not most derived
void base_ref(Base &b) { typeid(b); }
// CHECK-LABEL: define {{.*}}void @_Z8base_refR4Base
// CHECK:       %vtable
// CHECK:       ret void

// Not most derived
void base_deref(Base *b) { typeid(*b); }
// CHECK-LABEL: define {{.*}}void @_Z10base_derefP4Base
// CHECK:       %vtable
// CHECK:       ret void

// Not most derived
void nonfinal_ref(NonFinal &d) { typeid(d); }
// CHECK-LABEL: define {{.*}}void @_Z12nonfinal_refR8NonFinal
// CHECK:       %vtable
// CHECK:       ret void

// Not most derived
void nonfinal_deref(NonFinal *d) { typeid(*d); }
// CHECK-LABEL: define {{.*}}void @_Z14nonfinal_derefP8NonFinal
// CHECK:       %vtable
// CHECK:       ret void
