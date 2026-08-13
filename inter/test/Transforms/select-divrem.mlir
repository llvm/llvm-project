// RUN: inter-opt %s --inter-expand-arithmetic --inter-select-to-machine | FileCheck %s

module {
  func.func @unsigned_divrem(%divisor: i64) attributes {
      xemachine.kernel,
      xemachine.kernel_args = [
        #xemachine.kernel_arg<kind = by_value, address_space = "none", access = "none", size = 8, alignment = 8, offset = 24>
      ],
      xw.simd_width = 16 : i32} {
    %dividend = xw.global_id 0 : !xw.simd<i64, 16>
    %quotient = xw.binary divui %dividend, %divisor : !xw.simd<i64, 16>, i64 -> !xw.simd<i64, 16>
    %remainder = xw.binary remui %dividend, %divisor : !xw.simd<i64, 16>, i64 -> !xw.simd<i64, 16>
    return
  }
}

// -----

module {
  func.func @signed_divrem(%divisor: i64) attributes {
      xemachine.kernel,
      xemachine.kernel_args = [
        #xemachine.kernel_arg<kind = by_value, address_space = "none", access = "none", size = 8, alignment = 8, offset = 24>
      ],
      xw.simd_width = 16 : i32} {
    %dividend = xw.global_id 0 : !xw.simd<i64, 16>
    %quotient = xw.binary divsi %dividend, %divisor : !xw.simd<i64, 16>, i64 -> !xw.simd<i64, 16>
    %remainder = xw.binary remsi %dividend, %divisor : !xw.simd<i64, 16>, i64 -> !xw.simd<i64, 16>
    return
  }
}

// -----

module {
  func.func @bare_divrem() attributes {
      xemachine.kernel, xemachine.kernel_args = [],
      xw.simd_width = 8 : i32} {
    %positive = xw.constant 37 : i64
    %negative = xw.constant -37 : i64
    %divisor = xw.constant 5 : i64
    %udiv = xw.binary divui %positive, %divisor : i64, i64 -> i64
    %urem = xw.binary remui %positive, %divisor : i64, i64 -> i64
    %sdiv = xw.binary divsi %negative, %divisor : i64, i64 -> i64
    %srem = xw.binary remsi %negative, %divisor : i64, i64 -> i64
    return
  }
}

// -----

module {
  func.func @arith_i1() attributes {
      xemachine.kernel, xemachine.kernel_args = [],
      xw.simd_width = 8 : i32} {
    %true = arith.constant true
    %false = arith.xori %true, %true : i1
    %extended = arith.extui %false : i1 to i32
    scf.if %false {
    }
    return
  }
}

// CHECK-NOT: xw.binary
// CHECK-LABEL: func.func @unsigned_divrem
// CHECK: xemachine.shr
// CHECK: xemachine.cmp
// CHECK: xemachine.exec_if
// CHECK: xemachine.sub
// CHECK-LABEL: func.func @signed_divrem
// CHECK: xemachine.cmp {{.*}}signed
// CHECK: xemachine.exec_if
// CHECK: xemachine.shr
// CHECK: xemachine.sub
// CHECK-LABEL: func.func @bare_divrem
// CHECK: xemachine.shr
// CHECK: xemachine.cmp
// CHECK: xemachine.uniform_if
// CHECK: xemachine.sub
// CHECK-LABEL: func.func @arith_i1
// CHECK: %[[TRUE:.*]] = xemachine.imm -1 : i1
// CHECK: xemachine.or %[[TRUE]], %[[TRUE]]
// CHECK: xemachine.and %[[TRUE]], %[[TRUE]]
// CHECK: %[[FALSE:.*]] = xemachine.sub
// CHECK: xemachine.mov %[[FALSE]] {{.*}}src0Type = i1
// CHECK: %[[BRANCH:.*]] = xemachine.cmp ne %[[FALSE]]
// CHECK: xemachine.uniform_if %[[BRANCH]]
