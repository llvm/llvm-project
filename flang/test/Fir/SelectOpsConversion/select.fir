// RUN: fir-opt %s --fir-select-ops-conversion | FileCheck %s

// Exercises fir.select with an `index` selector and a mix of successors
// that forward one, two, or three block arguments, plus a unit default.

func.func @select(%arg : index, %arg2 : i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  fir.select %arg:index [ 1, ^bb1(%c1:i32),
                          2, ^bb2(%c3,%arg,%arg2:i32,index,i32),
                          3, ^bb3(%arg2,%c3:i32,i32),
                          4, ^bb4(%c2:i32),
                          unit, ^bb5 ]
  ^bb1(%a : i32) :
    return %a : i32
  ^bb2(%b : i32, %b2 : index, %b3:i32) :
    %castidx = arith.index_cast %b2 : index to i32
    %4 = arith.addi %b, %castidx : i32
    %5 = arith.addi %4, %b3 : i32
    return %5 : i32
  ^bb3(%c:i32, %c2b:i32) :
    %6 = arith.addi %c, %c2b : i32
    return %6 : i32
  ^bb4(%d : i32) :
    return %d : i32
  ^bb5 :
    %zero = arith.constant 0 : i32
    return %zero : i32
}

// CHECK-LABEL: func.func @select(
// CHECK-SAME:               %[[SELECTVALUE:.*]]: index,
// CHECK-SAME:               %[[ARG1:.*]]: i32)
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : i32
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : i32
// CHECK-DAG:     %[[C3:.*]] = arith.constant 3 : i32
// CHECK:         %[[SEL:.*]] = arith.index_cast %[[SELECTVALUE]] : index to i64
// CHECK:         cf.switch %[[SEL]] : i64, [
// CHECK:           default: ^[[BB5:.*]],
// CHECK:           1: ^[[BB1:.*]](%[[C1]] : i32),
// CHECK:           2: ^[[BB2:.*]](%[[C3]], %[[SELECTVALUE]], %[[ARG1]] : i32, index, i32),
// CHECK:           3: ^[[BB3:.*]](%[[ARG1]], %[[C3]] : i32, i32),
// CHECK:           4: ^[[BB4:.*]](%[[C2]] : i32)
// CHECK:         ]
// CHECK:       ^[[BB1]](%{{.*}}: i32):
// CHECK:         return
// CHECK:       ^[[BB2]](%{{.*}}: i32, %{{.*}}: index, %{{.*}}: i32):
// CHECK:         return
// CHECK:       ^[[BB3]](%{{.*}}: i32, %{{.*}}: i32):
// CHECK:         return
// CHECK:       ^[[BB4]](%{{.*}}: i32):
// CHECK:         return
// CHECK:       ^[[BB5]]:
// CHECK:         %[[CST0:.*]] = arith.constant 0 : i32
// CHECK:         return %[[CST0]] : i32

// -----

// Exercises the selector-widening code path across the four permitted
// selector types (i8, i16, i64, index) plus a case value that doesn't fit
// in i32. i8/i16 selectors are extended with `arith.extsi`, i64 is used
// as-is, and an `index` selector is bridged via `arith.index_cast`.

func.func @select_with_cast(%arg1 : i8, %arg2 : i16, %arg3: i64, %arg4: index) -> () {
  fir.select %arg1 : i8 [ 1, ^bb1, unit, ^bb1 ]
  ^bb1:
  fir.select %arg2 : i16 [ 1, ^bb2, unit, ^bb2 ]
  ^bb2:
  fir.select %arg3 : i64 [ 1, ^bb3, unit, ^bb3 ]
  ^bb3:
  fir.select %arg4 : index [ 1, ^bb4, unit, ^bb4 ]
  ^bb4:
  fir.select %arg3 : i64 [ 4294967296, ^bb5, unit, ^bb5 ]
  ^bb5:
  return
}

// CHECK-LABEL:   func.func @select_with_cast(
// CHECK-SAME:      %[[ARG0:.*]]: i8,
// CHECK-SAME:      %[[ARG1:.*]]: i16,
// CHECK-SAME:      %[[ARG2:.*]]: i64,
// CHECK-SAME:      %[[ARG3:.*]]: index)
// CHECK:           %[[V0:.*]] = arith.extsi %[[ARG0]] : i8 to i64
// CHECK:           cf.switch %[[V0]] : i64, [
// CHECK:             default: ^bb1,
// CHECK:             1: ^bb1
// CHECK:           ]
// CHECK:         ^bb1:
// CHECK:           %[[V1:.*]] = arith.extsi %[[ARG1]] : i16 to i64
// CHECK:           cf.switch %[[V1]] : i64, [
// CHECK:             default: ^bb2,
// CHECK:             1: ^bb2
// CHECK:           ]
// CHECK:         ^bb2:
// CHECK:           cf.switch %[[ARG2]] : i64, [
// CHECK:             default: ^bb3,
// CHECK:             1: ^bb3
// CHECK:           ]
// CHECK:         ^bb3:
// CHECK:           %[[V3:.*]] = arith.index_cast %[[ARG3]] : index to i64
// CHECK:           cf.switch %[[V3]] : i64, [
// CHECK:             default: ^bb4,
// CHECK:             1: ^bb4
// CHECK:           ]
// CHECK:         ^bb4:
// CHECK:           cf.switch %[[ARG2]] : i64, [
// CHECK:             default: ^bb5,
// CHECK:             4294967296: ^bb5
// CHECK:           ]
// CHECK:         ^bb5:
// CHECK:           return
