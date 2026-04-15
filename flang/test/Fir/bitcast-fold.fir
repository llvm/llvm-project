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
