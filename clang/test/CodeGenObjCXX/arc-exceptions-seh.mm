// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -emit-llvm -fobjc-arc -fexceptions -fobjc-exceptions -fobjc-arc-exceptions -fobjc-runtime=gnustep-2.0 -o - %s | FileCheck %s

// WinEH requires funclet tokens on nounwind intrinsics if they can lower to
// regular function calls in the course of IR transformations.
//
// This is the case for ObjC ARC runtime intrinsics. Test that clang emits the
// funclet tokens for llvm.objc.* calls inside catch- and cleanup-pads and that
// they refer to their pad's SSA value.

void do_something();
void may_throw(id);

void try_catch_with_objc_intrinsic() {
  id ex;
  @try {
    may_throw(ex);
  } @catch (id ex_caught) {
    do_something();
    may_throw(ex_caught);
  }
}

// CHECK-LABEL:   try_catch_with_objc_intrinsic
//
// CHECK:         catch.dispatch:
// CHECK-NEXT:      [[CATCHSWITCH:%[0-9]+]] = catchswitch within none [label %catch] unwind label %[[CLEANUP1:.*]]
//
// All calls within a catchpad must have funclet tokens that refer to it:
// CHECK:         catch:
// CHECK-NEXT:      [[CATCHPAD:%[0-9]+]] = catchpad within [[CATCHSWITCH]]
// CHECK:           call
// CHECK:             @llvm.objc.retain
// CHECK:             [ "funclet"(token [[CATCHPAD]]) ]
// CHECK:           invoke
// CHECK:             do_something
// CHECK:             [ "funclet"(token [[CATCHPAD]]) ]
// CHECK:             unwind label %[[CLEANUP2:.*]]
// CHECK:           invoke
// CHECK:             may_throw
// CHECK:             [ "funclet"(token [[CATCHPAD]]) ]
// CHECK:             unwind label %[[CLEANUP2]]
// CHECK:           call
// CHECK:             @llvm.objc.storeStrong
// CHECK:             [ "funclet"(token [[CATCHPAD]]) ]
// CHECK:           catchret from [[CATCHPAD]] to label %catchret.dest
//
// This block exists and it's empty:
// CHECK:         catchret.dest:
// CHECK-NEXT:      br label %eh.cont
//
// CHECK:         [[CLEANUP2]]:
// CHECK-NEXT:      [[CLEANUPPAD2:%[0-9]+]] = cleanuppad within [[CATCHPAD]]
// CHECK:           call
// CHECK:             @llvm.objc.storeStrong
// CHECK:             [ "funclet"(token [[CLEANUPPAD2]]) ]
// CHECK:           cleanupret from [[CLEANUPPAD2]]
// CHECK:             unwind label %[[CLEANUP1]]
//
// CHECK:         [[CLEANUP1]]:
// CHECK-NEXT:      [[CLEANUPPAD1:%[0-9]+]] = cleanuppad within none
// CHECK:           call
// CHECK:             @llvm.objc.storeStrong
// CHECK:             [ "funclet"(token [[CLEANUPPAD1]]) ]
// CHECK:           cleanupret from [[CLEANUPPAD1]] unwind to caller
