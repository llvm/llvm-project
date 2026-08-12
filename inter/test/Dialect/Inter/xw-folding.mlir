// RUN: inter-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @integer
// CHECK-NEXT: %[[C:.*]] = xw.constant 7 : i32
// CHECK-NEXT: return %[[C]] : i32
func.func @integer() -> i32 {
  %seven = xw.constant 7 : i32
  %zero = xw.constant 0 : i32
  %sum = xw.binary addi %seven, %zero : i32, i32 -> i32
  return %sum : i32
}

// CHECK-LABEL: func.func @select
// CHECK: return %arg0 : i32
func.func @select(%arg0: i32) -> i32 {
  %true = xw.constant true
  %other = xw.constant 9 : i32
  %selected = xw.select %true, %arg0, %other : i1, i32
  return %selected : i32
}

// CHECK-LABEL: func.func @read_splat
// CHECK: return %arg0 : i32
func.func @read_splat(%arg0: i32) -> i32 attributes {xw.simd_width = 32 : i64} {
  %splat = xw.splat %arg0 : i32 -> !xw.simd<i32, 8>
  %first = xw.read_first %splat : !xw.simd<i32, 8> -> i32
  return %first : i32
}

// CHECK-LABEL: func.func @token
// CHECK: return %[[TOKEN:.*]] : !xw.mem.token
func.func @token() -> !xw.mem.token {
  %root = xw.token : !xw.mem.token
  %joined = xw.join %root : !xw.mem.token -> !xw.mem.token
  return %joined : !xw.mem.token
}
