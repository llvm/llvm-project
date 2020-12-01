// Check the layout of the vtable for a normal class.
// The Fuchsia relative vtables ABI will be hidden behind a flag for now as part
// of a soft incremental rollout. This ABI should only be used if the flag for
// it is passed on Fuchsia.

// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -S -o - -emit-llvm -fexperimental-relative-c++-abi-vtables | FileCheck --check-prefix=RELATIVE-ABI %s
// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -S -o - -emit-llvm -fno-experimental-relative-c++-abi-vtables | FileCheck --check-prefix=DEFAULT-ABI %s
// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -S -o - -emit-llvm | FileCheck --check-prefix=DEFAULT-ABI %s

// VTable contains offsets and references to the hidden symbols
// RELATIVE-ABI: @_ZTV1A.local = private unnamed_addr constant { [3 x i32] } { [3 x i32] [i32 0, i32 trunc (i64 sub (i64 ptrtoint ({ i8*, i8* }** @_ZTI1A.rtti_proxy to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [3 x i32] }, { [3 x i32] }* @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (void (%class.A*)* dso_local_equivalent @_ZN1A3fooEv to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [3 x i32] }, { [3 x i32] }* @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32)] }, align 4
// RELATIVE-ABI: @_ZTV1A = unnamed_addr alias { [3 x i32] }, { [3 x i32] }* @_ZTV1A.local
// DEFAULT-ABI: @_ZTV1A = unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTI1A to i8*), i8* bitcast (void (%class.A*)* @_ZN1A3fooEv to i8*)] }, align 8

class A {
public:
  virtual void foo();
};

void A::foo() {}

void A_foo(A *a) {
  a->foo();
}
