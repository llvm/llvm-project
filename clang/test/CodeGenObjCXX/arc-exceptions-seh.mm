// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -emit-llvm -fobjc-arc -fexceptions -fobjc-exceptions -fobjc-runtime=gnustep-2.0 -o - %s | FileCheck %s

// WinEH requires funclet tokens on nounwind intrinsics if they can lower to
// regular function calls in the course of IR transformations.
//
// This is the case for ObjC ARC runtime intrinsics. Test that clang emits the
// funclet tokens for llvm.objc.retain and llvm.objc.storeStrong and that they
// refer to their catchpad's SSA value.

@class Ety;
void opaque(void);
void test_catch_with_objc_intrinsic(void) {
  @try {
    opaque();
  } @catch (Ety *ex) {
    // Destroy ex when leaving catchpad. This emits calls to intrinsic functions
    // llvm.objc.retain and llvm.objc.storeStrong
  }
}

// CHECK-LABEL: define{{.*}} void {{.*}}test_catch_with_objc_intrinsic
//                ...
// CHECK:       catch.dispatch:
// CHECK-NEXT:    [[CATCHSWITCH:%[0-9]+]] = catchswitch within none
//                ...
// CHECK:       catch:
// CHECK-NEXT:    [[CATCHPAD:%[0-9]+]] = catchpad within [[CATCHSWITCH]]
// CHECK:         {{%[0-9]+}} = call {{.*}} @llvm.objc.retain{{.*}} [ "funclet"(token [[CATCHPAD]]) ]
// CHECK:         call {{.*}} @llvm.objc.storeStrong{{.*}} [ "funclet"(token [[CATCHPAD]]) ]
