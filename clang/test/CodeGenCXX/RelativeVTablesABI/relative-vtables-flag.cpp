// Check the layout of the vtable for a normal class.
// The Fuchsia relative vtables ABI will be hidden behind a flag for now as part
// of a soft incremental rollout. This ABI should only be used if the flag for
// it is passed on Fuchsia.

// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -o - -emit-llvm -fhalf-no-semantic-interposition | FileCheck --check-prefix=RELATIVE-ABI %s
// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -o - -emit-llvm -fno-experimental-relative-c++-abi-vtables | FileCheck --check-prefix=DEFAULT-ABI %s

// VTable contains offsets and references to the hidden symbols
// RELATIVE-ABI: @_ZTV1A.local = internal unnamed_addr constant { [3 x i32] } { [3 x i32] [i32 0, i32 trunc (i64 sub (i64 ptrtoint (ptr @_ZTI1A.rtti_proxy to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [3 x i32] }, ptr @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1A3fooEv to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [3 x i32] }, ptr @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32)] }, align 4
// RELATIVE-ABI: @_ZTV1A ={{.*}} unnamed_addr alias { [3 x i32] }, ptr @_ZTV1A.local
// DEFAULT-ABI: @_ZTV1A ={{.*}} unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI1A, ptr @_ZN1A3fooEv] }, align 8

class A {
public:
  virtual void foo();
};

void A::foo() {}

void A_foo(A *a) {
  a->foo();
}
