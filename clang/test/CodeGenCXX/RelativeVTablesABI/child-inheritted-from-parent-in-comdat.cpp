// Cross comdat example
// Parent VTable is in a comdat section.

// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -O1 -S -o - -emit-llvm -fhalf-no-semantic-interposition | FileCheck %s

// The inline function is emitted in each module with the same comdat
// CHECK: $_ZTS1A = comdat any
// CHECK: $_ZTI1A = comdat any
// CHECK: $_ZTI1B.rtti_proxy = comdat any

// The VTable is emitted everywhere used
// CHECK: $_ZTV1A = comdat any
// CHECK: $_ZTI1A.rtti_proxy = comdat any

// The VTable for B is emitted here since it has a key function which is defined in this module
// CHECK: @_ZTV1B.local = private unnamed_addr constant { [3 x i32] } { [3 x i32] [i32 0, i32 trunc (i64 sub (i64 ptrtoint (ptr @_ZTI1B.rtti_proxy to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [3 x i32] }, ptr @_ZTV1B.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1B3fooEv to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [3 x i32] }, ptr @_ZTV1B.local, i32 0, i32 0, i32 2) to i64)) to i32)] }, align 4

// The VTable for A is emitted here and in a comdat section since it has no key function, and is used in this module when creating an instance of A (in func()).
// CHECK: @_ZTV1A.local = linkonce_odr hidden unnamed_addr constant { [3 x i32] } { [3 x i32] [i32 0, i32 trunc (i64 sub (i64 ptrtoint (ptr @_ZTI1A.rtti_proxy to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [3 x i32] }, ptr @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1A3fooEv to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [3 x i32] }, ptr @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32)] }, comdat($_ZTV1A), align 4

// CHECK: @_ZTV1B ={{.*}} unnamed_addr alias { [3 x i32] }, ptr @_ZTV1B.local
// CHECK: @_ZTV1A = linkonce_odr unnamed_addr alias { [3 x i32] }, ptr @_ZTV1A.local

// CHECK:      define{{.*}} void @_ZN1B3fooEv(ptr {{.*}}%this) unnamed_addr
// CHECK-NEXT: entry:
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

class A {
public:
  inline virtual void foo() {}
};
class B : public A {
public:
  void foo() override;
};
void A_foo(A *a);

void B::foo() {}
void func2() {
  A a;
  A_foo(&a);
}
