// RUN: fir-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: @bitcast_roundtrip_i32_logical
func.func @bitcast_roundtrip_i32_logical(%x : i32) -> i32 {
  %0 = fir.bitcast %x : (i32) -> !fir.logical<4>
  %1 = fir.bitcast %0 : (!fir.logical<4>) -> i32
  // CHECK-NEXT: return %{{.*}} : i32
  return %1 : i32
}

// CHECK-LABEL: @bitcast_roundtrip_f32_i32
func.func @bitcast_roundtrip_f32_i32(%x : f32) -> f32 {
  %0 = fir.bitcast %x : (f32) -> i32
  %1 = fir.bitcast %0 : (i32) -> f32
  // CHECK-NEXT: return %{{.*}} : f32
  return %1 : f32
}

// CHECK-LABEL: @bitcast_chain_collapse
func.func @bitcast_chain_collapse(%x : i32) -> !fir.logical<4> {
  %0 = fir.bitcast %x : (i32) -> f32
  %1 = fir.bitcast %0 : (f32) -> !fir.logical<4>
  // CHECK-NEXT: %[[RES:.*]] = fir.bitcast %{{.*}} : (i32) -> !fir.logical<4>
  // CHECK-NEXT: return %[[RES]]
  return %1 : !fir.logical<4>
}

func.func @bitcast_convert_fold_true() -> i1 {
  %c3_i32 = arith.constant 3 : i32
  %1 = fir.bitcast %c3_i32 : (i32) -> !fir.logical<4>
  %2 = fir.convert %1 : (!fir.logical<4>) -> i1
  return %2 : i1
}
// CHECK-LABEL:   func.func @bitcast_convert_fold_true() -> i1 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant true
// CHECK:           return %[[CONSTANT_0]] : i1

func.func @bitcast_convert_fold_false() -> i1 {
  %c0_i32 = arith.constant 0 : i32
  %1 = fir.bitcast %c0_i32 : (i32) -> !fir.logical<4>
  %2 = fir.convert %1 : (!fir.logical<4>) -> i1
  return %2 : i1
}
// CHECK-LABEL:   func.func @bitcast_convert_fold_false() -> i1 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant false
// CHECK:           return %[[CONSTANT_0]] : i1
