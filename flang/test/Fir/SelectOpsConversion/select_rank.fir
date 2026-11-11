// RUN: fir-opt %s --fir-select-ops-conversion | FileCheck %s

// Exercises fir.select_rank with an i32 selector and successors that
// forward one, two, or three block arguments, plus a unit default. The
// rank selector is widened to i64 with `arith.extsi` before `cf.switch`.

func.func @select_rank(%arg : i32, %arg2 : i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  fir.select_rank %arg:i32 [ 1, ^bb1(%c1:i32),
                             2, ^bb2(%c3,%arg,%arg2:i32,i32,i32),
                             3, ^bb3(%arg2,%c3:i32,i32),
                             4, ^bb4(%c2:i32),
                             unit, ^bb5 ]
  ^bb1(%a : i32) :
    return %a : i32
  ^bb2(%b : i32, %b2 : i32, %b3:i32) :
    %4 = arith.addi %b, %b2 : i32
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

// CHECK-LABEL: func.func @select_rank(
// CHECK-SAME:                    %[[SELECTVALUE:.*]]: i32,
// CHECK-SAME:                    %[[ARG1:.*]]: i32)
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : i32
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : i32
// CHECK-DAG:     %[[C3:.*]] = arith.constant 3 : i32
// CHECK:         %[[SELECTOR:.*]] = arith.extsi %[[SELECTVALUE]] : i32 to i64
// CHECK:         cf.switch %[[SELECTOR]] : i64, [
// CHECK:           default: ^{{.*}},
// CHECK:           1: ^{{.*}}(%[[C1]] : i32),
// CHECK:           2: ^{{.*}}(%[[C3]], %[[SELECTVALUE]], %[[ARG1]] : i32, i32, i32),
// CHECK:           3: ^{{.*}}(%[[ARG1]], %[[C3]] : i32, i32),
// CHECK:           4: ^{{.*}}(%[[C2]] : i32)
// CHECK:         ]
