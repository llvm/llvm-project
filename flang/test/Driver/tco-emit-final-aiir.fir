// Test tco's -emit-final-aiir option.

// RUN: tco -emit-final-aiir %s 2>&1 | FileCheck %s

// Check that the FIR file is translated into the LLVM IR
// AIIR dialect, but not into LLVM IR itself.

// CHECK-NOT: func.func
// CHECK-LABEL: llvm.func @_QPfoo()
// CHECK-NOT: fir.alloca
// CHECK: %[[VAL_0:.*]] = llvm.aiir.constant(1 : i64) : i64
// CHECK: %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x i32 : (i64) -> !llvm.ptr
// CHECK: llvm.return
// CHECK-NOT: func.func

func.func @_QPfoo() {
  %1 = fir.alloca i32
  %0 = arith.constant 0 : i32
  fir.store %0 to %1 : !fir.ref<i32>
  return
}
