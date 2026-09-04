// RUN: inter-opt %s --inter-convert-llvm-to-xw | FileCheck %s

module {
  func.func @freeze() attributes {xw.simd_width = 16 : i64} {
    %lane = xw.lane_id : !xw.simd<i32, 16>
    %cast = builtin.unrealized_conversion_cast %lane
        : !xw.simd<i32, 16> to i32
    %frozen = xw.freeze %cast : i32
    return
  }

  // CHECK-LABEL: func.func @freeze
  // CHECK: %[[LANE:.*]] = xw.lane_id : !xw.simd<i32, 16>
  // CHECK: xw.freeze %[[LANE]] : !xw.simd<i32, 16>
  // CHECK-NOT: unrealized_conversion_cast

  func.func @if_result(%condition: i1, %scalar: i32)
      attributes {xw.simd_width = 16 : i64} {
    %lane = xw.lane_id : !xw.simd<i32, 16>
    %cast = builtin.unrealized_conversion_cast %lane
        : !xw.simd<i32, 16> to i32
    %result = scf.if %condition -> i32 {
      scf.yield %cast : i32
    } else {
      scf.yield %scalar : i32
    }
    return
  }

  // CHECK-LABEL: func.func @if_result
  // CHECK: %[[IF:.*]] = scf.if {{.*}} -> (!xw.simd<i32, 16>) {
  // CHECK: scf.yield {{.*}} : !xw.simd<i32, 16>
  // CHECK: } else {
  // CHECK: %[[IF_SPLAT:.*]] = xw.splat {{.*}} : i32 -> !xw.simd<i32, 16>
  // CHECK: scf.yield %[[IF_SPLAT]] : !xw.simd<i32, 16>

  func.func @resultless_if(%condition: i1) {
    scf.if %condition {
      %token = xw.token : !xw.mem.token
    }
    return
  }

  // CHECK-LABEL: func.func @resultless_if
  // CHECK: scf.if %{{.*}} {
  // CHECK: xw.token

  func.func @for_iter_arg(%scalar: i32)
      attributes {xw.simd_width = 16 : i64} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %lane = xw.lane_id : !xw.simd<i32, 16>
    %cast = builtin.unrealized_conversion_cast %lane
        : !xw.simd<i32, 16> to i32
    %result = scf.for %iv = %zero to %one step %one
        iter_args(%iter = %scalar) -> i32 {
      scf.yield %cast : i32
    }
    return
  }

  // CHECK-LABEL: func.func @for_iter_arg
  // CHECK: %[[FOR_INIT:.*]] = xw.splat {{.*}} : i32 -> !xw.simd<i32, 16>
  // CHECK: scf.for {{.*}} iter_args(%{{.*}} = %[[FOR_INIT]]) -> (!xw.simd<i32, 16>)
  // CHECK: scf.yield {{.*}} : !xw.simd<i32, 16>

  func.func @while_loop_carried(%scalar: i32, %condition: i1)
      attributes {xw.simd_width = 16 : i64} {
    %lane = xw.lane_id : !xw.simd<i32, 16>
    %cast = builtin.unrealized_conversion_cast %lane
        : !xw.simd<i32, 16> to i32
    %result = scf.while (%iter = %scalar) : (i32) -> i32 {
      scf.condition(%condition) %cast : i32
    } do {
    ^bb0(%after: i32):
      scf.yield %after : i32
    }
    return
  }

  // CHECK-LABEL: func.func @while_loop_carried
  // CHECK: %[[WHILE_INIT:.*]] = xw.splat {{.*}} : i32 -> !xw.simd<i32, 16>
  // CHECK: scf.while (%{{.*}} = %[[WHILE_INIT]]) : (!xw.simd<i32, 16>) -> !xw.simd<i32, 16> {
  // CHECK: scf.condition({{.*}}) {{.*}} : !xw.simd<i32, 16>
  // CHECK: ^bb0(%{{.*}}: !xw.simd<i32, 16>)
  // CHECK: scf.yield {{.*}} : !xw.simd<i32, 16>
}
