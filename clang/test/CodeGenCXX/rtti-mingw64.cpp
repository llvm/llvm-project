// RUN: %clang_cc1 -triple x86_64-windows-gnu %s -emit-llvm -o - | FileCheck %s
struct A { int a; };
struct B : virtual A { int b; };
B b;
class C {
  virtual ~C();
};
C::~C() {}

// CHECK: @_ZTI1C = linkonce_odr dso_local
// CHECK: @_ZTI1B = linkonce_odr dso_local constant { ptr, ptr, i32, i32, ptr, i64 }
// CHECK-SAME:  ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv121__vmi_class_type_infoE, i64 2),
// CHECK-SAME:  ptr @_ZTS1B,
// CHECK-SAME:  i32 0,
// CHECK-SAME:  i32 1,
// CHECK-SAME:  ptr @_ZTI1A,
//    This i64 is important, it should be an i64, not an i32.
// CHECK-SAME:  i64 -6141 }, comdat
