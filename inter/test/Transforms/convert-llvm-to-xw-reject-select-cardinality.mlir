// RUN: not inter-opt %s --inter-convert-llvm-to-xw --verify-each=false \
// RUN:   2>&1 | FileCheck %s

module {
  func.func @bad(%condition: !xw.mask<16>, %lhs: !xw.simd<i32, 8>,
                 %rhs: !xw.simd<i32, 8>) {
    %result = "xw.select"(%condition, %lhs, %rhs)
        : (!xw.mask<16>, !xw.simd<i32, 8>, !xw.simd<i32, 8>)
          -> !xw.simd<i32, 8>
    return
  }
}

// CHECK: mask condition requires a matching SIMD or mask result
