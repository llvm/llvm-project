// RUN: inter-opt %s --inter-select-to-machine | FileCheck %s --check-prefix=SELECT
// RUN: inter-opt %s --inter-select-to-machine --inter-prepare-regalloc | FileCheck %s --check-prefix=PREP

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

// SELECT-NOT: llvm
// SELECT: xemachine.mov {{.*}}execSize = 32
// SELECT-SAME: src0Region = #xemachine.region<1, 4, 0>
// SELECT: xemachine.cmp ne %[[LANES:.*]], %[[LANES]] {{.*}}execSize = 8
// SELECT: %[[PREDICATED:.*]] = xemachine.exec_if
// SELECT: %[[SUM:.*]] = xemachine.add
// SELECT-NEXT: xemachine.yield %[[SUM]]
// SELECT: otherwise {
// SELECT-NEXT: xemachine.yield %[[LANES]]
// SELECT: xemachine.cmp {{.*}}execSize = 1
// SELECT: xemachine.uniform_if
// SELECT: xemachine.yield %[[PREDICATED]]
// SELECT: xemachine.yield %[[LANES]]
// SELECT: %[[R0:.*]] = xemachine.archreg 0
// SELECT-NEXT: xemachine.eot %[[R0]] : !xemachine.reg<16, 0>

// PREP: xemachine.exec_if
// PREP: xemachine.mov {{.*}}xemachine.regalloc_copy = "branch-yield"
// PREP: xemachine.yield
