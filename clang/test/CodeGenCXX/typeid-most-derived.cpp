// RUN: %clang_cc1 %s -triple %itanium_abi_triple -Wno-unused-value -emit-llvm -o - -std=c++11 | FileCheck %s

namespace std {
  class type_info {};
}

struct Poly {
  virtual int foo() { return 42; }
  virtual ~Poly();
};

struct Derived : Poly {};
struct Final final : Poly {
    int foo() override { return 84; }
};

// Most derived
void value(Poly p) { typeid(p); }
// CHECK-LABEL: define {{.*}}void @_Z5value4Poly
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
void poly_ref(Poly &p) { typeid(p); }
// CHECK-LABEL: define {{.*}}void @_Z8poly_refR4Poly
// CHECK:       %vtable
// CHECK:       ret void

// Not most derived
void poly_deref(Poly *p) { typeid(*p); }
// CHECK-LABEL: define {{.*}}void @_Z10poly_derefP4Poly
// CHECK:       %vtable
// CHECK:       ret void

// Not most derived
void derived_ref(Derived &d) { typeid(d); }
// CHECK-LABEL: define {{.*}}void @_Z11derived_refR7Derived
// CHECK:       %vtable
// CHECK:       ret void

// Not most derived
void derived_deref(Derived *d) { typeid(*d); }
// CHECK-LABEL: define {{.*}}void @_Z13derived_derefP7Derived
// CHECK:       %vtable
// CHECK:       ret void
