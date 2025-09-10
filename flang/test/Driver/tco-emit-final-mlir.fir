// Test tco's -emit-final-mlir option.

// RUN: tco -emit-final-mlir %s 2>&1 | FileCheck %s

// Check that the FIR file is translated into the LLVM IR
// MLIR dialect, but not into LLVM IR itself.

// CHECK-NOT: func.func
// CHECK-LABEL: llvm.func @_QPfoo()
// CHECK-NOT: fir.alloca
// CHECK: %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x i32 : (i64) -> !llvm.ptr
// CHECK: llvm.return
// CHECK-NOT: func.func

func.func @_QPfoo() {
  %1 = fir.alloca i32
  return
}
