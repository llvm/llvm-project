// RUN: inter-opt %s --inter-select-to-machine | FileCheck %s

module {
  func.func @control() attributes {
      xemachine.kernel, xemachine.kernel_args = [],
      xw.simd_width = 32 : i32} {
    %zero = xw.constant 0 : i32
    %one = xw.constant 1 : i32
    %lanes = xw.splat %zero : i32 -> !xw.simd<i32, 8>
    %expanded = xw.expand %lanes : !xw.simd<i32, 8> -> !xw.simd<i32, 32>
    %expanded_first = xw.read_first %expanded : !xw.simd<i32, 32> -> i32
    %mask = xw.cmpi ne %lanes, %lanes : !xw.simd<i32, 8>, !xw.simd<i32, 8> -> !xw.mask<8>
    %predicated = xw.where %mask {
      %sum = xw.binary addi %lanes, %one : !xw.simd<i32, 8>, i32 -> !xw.simd<i32, 8>
      xw.yield %sum : !xw.simd<i32, 8>
    } otherwise {
      xw.yield %lanes : !xw.simd<i32, 8>
    } : !xw.mask<8> -> !xw.simd<i32, 8>
    %uniform = xw.cmpi eq %expanded_first, %one : i32, i32 -> i1
    %selected = scf.if %uniform -> (!xw.simd<i32, 8>) {
      scf.yield %predicated : !xw.simd<i32, 8>
    } else {
      scf.yield %lanes : !xw.simd<i32, 8>
    }
    return
  }
}

// CHECK-NOT: llvm
// CHECK: xemachine.mov {{.*}}execSize = 32
// CHECK-SAME: src0Region = #xemachine.region<1, 4, 0>
// CHECK: xemachine.cmp {{.*}}execSize = 8
// CHECK: xemachine.exec_if
// CHECK: xemachine.yield
// CHECK: xemachine.cmp {{.*}}execSize = 1
// CHECK: xemachine.uniform_if
