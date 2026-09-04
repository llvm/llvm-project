// RUN: not inter-opt %s --inter-convert-llvm-to-xw --verify-each=false \
// RUN:   2>&1 | FileCheck %s

module {
  func.func @bad(%lhs: !xw.simd<i32, 8>, %rhs: !xw.simd<i32, 16>) {
    %result = "xw.cmpi"(%lhs, %rhs) <{predicate = 0 : i64}>
        : (!xw.simd<i32, 8>, !xw.simd<i32, 16>) -> !xw.mask<8>
    return
  }
}

// CHECK: operands must have the same type
