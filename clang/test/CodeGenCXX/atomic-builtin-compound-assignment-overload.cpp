// RUN: %clang_cc1 -std=gnu++11 -emit-llvm -triple=x86_64-linux-gnu -o - %s | FileCheck %s

_Atomic unsigned an_atomic_uint;

enum { an_enum_value = 1 };

// CHECK-LABEL: define {{.*}}void @_Z5enum1v()
void enum1() {
  an_atomic_uint += an_enum_value;
  // CHECK: atomicrmw add ptr
}

// CHECK-LABEL: define {{.*}}void @_Z5enum2v()
void enum2() {
  an_atomic_uint |= an_enum_value;
  // CHECK: atomicrmw or ptr
}

// CHECK-LABEL: define {{.*}}void @_Z5enum3RU7_Atomicj({{.*}})
void enum3(_Atomic unsigned &an_atomic_uint_param) {
  an_atomic_uint_param += an_enum_value;
  // CHECK: atomicrmw add ptr
}

// CHECK-LABEL: define {{.*}}void @_Z5enum4RU7_Atomicj({{.*}})
void enum4(_Atomic unsigned &an_atomic_uint_param) {
  an_atomic_uint_param |= an_enum_value;
  // CHECK: atomicrmw or ptr
}

volatile _Atomic unsigned an_volatile_atomic_uint;

// CHECK-LABEL: define {{.*}}void @_Z5enum5v()
void enum5() {
  an_volatile_atomic_uint += an_enum_value;
  // CHECK: atomicrmw add ptr
}

// CHECK-LABEL: define {{.*}}void @_Z5enum6v()
void enum6() {
  an_volatile_atomic_uint |= an_enum_value;
  // CHECK: atomicrmw or ptr
}

// CHECK-LABEL: define {{.*}}void @_Z5enum7RVU7_Atomicj({{.*}})
void enum7(volatile _Atomic unsigned &an_volatile_atomic_uint_param) {
  an_volatile_atomic_uint_param += an_enum_value;
  // CHECK: atomicrmw add ptr
}

// CHECK-LABEL: define {{.*}}void @_Z5enum8RVU7_Atomicj({{.*}})
void enum8(volatile _Atomic unsigned &an_volatile_atomic_uint_param) {
  an_volatile_atomic_uint_param |= an_enum_value;
  // CHECK: atomicrmw or ptr
}
