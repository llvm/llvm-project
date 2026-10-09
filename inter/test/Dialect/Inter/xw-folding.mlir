// RUN: inter-opt -canonicalize -cse %s | FileCheck %s

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

// CHECK-LABEL: func.func @freeze
// CHECK: %[[POISON:.*]] = ub.poison : i32
// CHECK-NEXT: %[[A:.*]] = xw.freeze %[[POISON]] : i32
// CHECK-NEXT: %[[B:.*]] = xw.freeze %[[POISON]] : i32
// CHECK-NEXT: return %[[A]], %[[B]] : i32, i32
func.func @freeze() -> (i32, i32) {
  %poison = ub.poison : i32
  %a = xw.freeze %poison : i32
  %b = xw.freeze %poison : i32
  return %a, %b : i32, i32
}

// CHECK-LABEL: func.func @power_of_two
// CHECK: %[[MASK:.*]] = xw.constant 3 : i64
// CHECK: %[[SHIFT:.*]] = xw.constant 2 : i64
// CHECK: %[[DIV:.*]] = xw.binary shrui %arg0, %[[SHIFT]]
// CHECK: %[[REM:.*]] = xw.binary andi %arg0, %[[MASK]]
// CHECK: %[[MUL:.*]] = xw.binary shli %arg0, %[[SHIFT]] overflow<nsw>
// CHECK: return %[[DIV]], %[[REM]], %[[MUL]]
func.func @power_of_two(%arg0: i64) -> (i64, i64, i64) {
  %four = xw.constant 4 : i64
  %div = xw.binary divui %arg0, %four : i64, i64 -> i64
  %rem = xw.binary remui %arg0, %four : i64, i64 -> i64
  %mul = xw.binary muli %four, %arg0 overflow<nsw>
      : i64, i64 -> i64
  return %div, %rem, %mul : i64, i64, i64
}

// CHECK-LABEL: func.func @simd_power_of_two
// CHECK: xw.binary shrui
// CHECK: xw.binary andi
func.func @simd_power_of_two(%arg0: !xw.simd<i32, 16>)
    -> (!xw.simd<i32, 16>, !xw.simd<i32, 16>) attributes {
      xw.simd_width = 16 : i32} {
  %four = xw.constant 4 : i32
  %div = xw.binary divui %arg0, %four
      : !xw.simd<i32, 16>, i32 -> !xw.simd<i32, 16>
  %rem = xw.binary remui %arg0, %four
      : !xw.simd<i32, 16>, i32 -> !xw.simd<i32, 16>
  return %div, %rem : !xw.simd<i32, 16>, !xw.simd<i32, 16>
}

// CHECK-LABEL: func.func @integer_cast_chain
// CHECK: %[[EXTENDED:.*]] = xw.cast intconvert %arg0 policy
// CHECK-NEXT: %[[FLAGGED:.*]] = xw.cast intconvert %[[EXTENDED]] overflow<nuw>
// CHECK-NEXT: return %arg0, %[[FLAGGED]] : i32, i32
func.func @integer_cast_chain(%arg0: i32) -> (i32, i32) {
  %extended = xw.cast intconvert %arg0
      policy {extension = #xw.cast_extension<sign>} : i32 -> i64
  %plain = xw.cast intconvert %extended : i64 -> i32
  %flagged = xw.cast intconvert %extended overflow<nuw> : i64 -> i32
  return %plain, %flagged : i32, i32
}
