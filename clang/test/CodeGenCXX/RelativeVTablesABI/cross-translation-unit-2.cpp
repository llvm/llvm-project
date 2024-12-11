// Check the vtable layout for classes with key functions defined in different
// translation units. This TU manifests the vtable for B.

// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -O1 -o - -emit-llvm -fhalf-no-semantic-interposition | FileCheck %s

#include "cross-tu-header.h"

// CHECK: $_ZTI1B.rtti_proxy = comdat any

// CHECK: @_ZTV1B.local = internal unnamed_addr constant { [4 x i32] } { [4 x i32] [i32 0, i32 trunc (i64 sub (i64 ptrtoint (ptr @_ZTI1B.rtti_proxy to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [4 x i32] }, ptr @_ZTV1B.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1B3fooEv to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [4 x i32] }, ptr @_ZTV1B.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1A3barEv to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [4 x i32] }, ptr @_ZTV1B.local, i32 0, i32 0, i32 2) to i64)) to i32)] }, align 4
// CHECK: @_ZTV1B ={{.*}} unnamed_addr alias { [4 x i32] }, ptr @_ZTV1B.local

// A::bar() is defined outside of the module that defines the vtable for A
// CHECK:      define{{.*}} void @_ZN1A3barEv(ptr {{.*}}%this) unnamed_addr
// CHECK-NEXT: entry:
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

// CHECK:      define{{.*}} void @_ZN1B3fooEv(ptr {{.*}}%this) unnamed_addr
// CHECK-NEXT: entry:
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

void A::bar() {}
void B::foo() {}
