// RUN: %clang_cc1 -triple x86_64-windows-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64le-windows-gnu -emit-llvm -o - %s | FileCheck %s

// An empty struct is handled as a struct with a dummy i8, on all targets.
// Most targets treat an empty struct return value as essentially void - but
// some don't. (Currently, at least x86_64-windows-* and powerpc64le-* don't
// treat it as void.)
//
// When intializing a struct with such a no_unique_address member, make sure we
// don't write the dummy i8 into the struct where there's no space allocated for
// it.
//
// This can only be tested with targets that don't treat empty struct returns as
// void.

struct S {};
S f();
struct Z {
  int x;
  [[no_unique_address]] S y;
  Z();
};
Z::Z() : x(111), y(f()) {}

// CHECK: define {{.*}} @_ZN1ZC2Ev
// CHECK: %call = call i8 @_Z1fv()
// CHECK-NEXT: ret void


// Check that the constructor for an empty member gets called with the right
// 'this' pointer.

struct S2 {
  S2();
};
struct Z2 {
  int x;
  [[no_unique_address]] S2 y;
  Z2();
};
Z2::Z2() : x(111) {}

// CHECK: define {{.*}} @_ZN2Z2C2Ev(ptr {{.*}} %this)
// CHECK: %this.addr = alloca ptr
// CHECK: store ptr %this, ptr %this.addr
// CHECK: %this1 = load ptr, ptr %this.addr
// CHECK: call void @_ZN2S2C1Ev(ptr {{.*}} %this1)
