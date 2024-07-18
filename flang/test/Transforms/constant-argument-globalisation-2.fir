// RUN: fir-opt --split-input-file --constant-argument-globalisation-opt < %s | FileCheck %s

module {
// Test for "two conditional writes to the same alloca doesn't get replaced."
  func.func @func(%arg0: i32, %arg1: i1) {
    %c2_i32 = arith.constant 2 : i32
    %addr = fir.alloca i32 {adapt.valuebyref}
    fir.if %arg1 {
      fir.store %c2_i32 to %addr : !fir.ref<i32>
    } else {
      fir.store %arg0 to %addr : !fir.ref<i32>
    }
    fir.call @sub2(%addr) : (!fir.ref<i32>) -> ()
    return
  }
  func.func private @sub2(!fir.ref<i32>)

// CHECK-LABEL: func.func @func
// CHECK-SAME: [[ARG0:%.*]]: i32
// CHECK-SAME: [[ARG1:%.*]]: i1)
// CHECK: [[CONST:%.*]] = arith.constant
// CHECK: [[ADDR:%.*]] = fir.alloca i32
// CHECK: fir.if [[ARG1]]
// CHECK: fir.store [[CONST]] to [[ADDR]]
// CHECK:  } else {
// CHECK: fir.store [[ARG0]] to [[ADDR]]
// CHECK: fir.call @sub2([[ADDR]])
// CHECK: return

}

// -----

module {
// Test for "two writes to the same alloca doesn't get replaced."
  func.func @func() {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %addr = fir.alloca i32 {adapt.valuebyref}
    fir.store %c1_i32 to %addr : !fir.ref<i32>
    fir.store %c2_i32 to %addr : !fir.ref<i32>
    fir.call @sub2(%addr) : (!fir.ref<i32>) -> ()
    return
  }
  func.func private @sub2(!fir.ref<i32>)

// CHECK-LABEL: func.func @func
// CHECK: [[CONST1:%.*]] = arith.constant
// CHECK: [[CONST2:%.*]] = arith.constant
// CHECK: [[ADDR:%.*]] = fir.alloca i32
// CHECK: fir.store [[CONST1]] to [[ADDR]]
// CHECK: fir.store [[CONST2]] to [[ADDR]]
// CHECK: fir.call @sub2([[ADDR]])
// CHECK: return

}

// -----

module {
// Test for "one write to the the alloca gets replaced."
  func.func @func() {
    %c1_i32 = arith.constant 1 : i32
    %addr = fir.alloca i32 {adapt.valuebyref}
    fir.store %c1_i32 to %addr : !fir.ref<i32>
    fir.call @sub2(%addr) : (!fir.ref<i32>) -> ()
    return
  }
  func.func private @sub2(!fir.ref<i32>)

// CHECK-LABEL: func.func @func
// CHECK: [[ADDR:%.*]] = fir.address_of([[EXTR:@.*]]) : !fir.ref<i32>
// CHECK: fir.call @sub2([[ADDR]])
// CHECK: return
// CHECK: fir.global internal [[EXTR]] constant : i32 {
// CHECK: %{{.*}} = arith.constant 1 : i32
// CHECK: fir.has_value %{{.*}} : i32
// CHECK: }

}

// -----
// Check that same argument used twice is converted.
module {
  func.func @func(%arg0: !fir.ref<i32>, %arg1: i1) {
    %c2_i32 = arith.constant 2 : i32
    %addr1 = fir.alloca i32 {adapt.valuebyref}
    fir.store %c2_i32 to %addr1 : !fir.ref<i32>
    fir.call @sub1(%addr1, %addr1) : (!fir.ref<i32>, !fir.ref<i32>) -> ()
    return
  }
}

// CHECK-LABEL: func.func @func
// CHECK-NEXT: %[[ARG1:.*]] = fir.address_of([[CONST1:@.*]]) : !fir.ref<i32>
// CHECK-NEXT: %[[ARG2:.*]] = fir.address_of([[CONST2:@.*]]) : !fir.ref<i32>
// CHECK-NEXT: fir.call @sub1(%[[ARG1]], %[[ARG2]])
// CHECK-NEXT: return
