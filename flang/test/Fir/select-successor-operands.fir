// RUN: fir-opt %s --canonicalize="region-simplify=aggressive" | FileCheck %s
// RUN: fir-opt %s --canonicalize="region-simplify=aggressive" --mlir-print-op-generic | fir-opt | FileCheck %s

// Removing an unused block argument must update the operand count of each
// incoming edge, including multiple edges from the same operation.
// CHECK-LABEL: func.func @select_remove_unused(
// CHECK-SAME: %[[SEL:.*]]: i32, %[[A:.*]]: i32, %[[B:.*]]: i32)
// CHECK: fir.select %[[SEL]] : i32 [1, ^[[DEST:bb[0-9]+]](%[[A]] : i32), unit, ^[[DEST]](%[[B]] : i32)]
// CHECK: ^[[DEST]](%[[VALUE:.*]]: i32):
// CHECK-NEXT: return %[[VALUE]] : i32
func.func @select_remove_unused(%selector: i32, %a: i32, %b: i32) -> i32 {
  fir.select %selector : i32 [1, ^dest(%a, %b : i32, i32),
                             unit, ^dest(%b, %a : i32, i32)]
^dest(%value: i32, %unused: i32):
  return %value : i32
}

// CHECK-LABEL: func.func @select_rank_remove_unused(
// CHECK-SAME: %[[SEL:.*]]: i32, %[[A:.*]]: i32, %[[B:.*]]: i32)
// CHECK: fir.select_rank %[[SEL]] : i32 [1, ^[[DEST:bb[0-9]+]](%[[A]] : i32), unit, ^[[DEST]](%[[B]] : i32)]
// CHECK: ^[[DEST]](%[[VALUE:.*]]: i32):
// CHECK-NEXT: return %[[VALUE]] : i32
func.func @select_rank_remove_unused(%selector: i32, %a: i32, %b: i32) -> i32 {
  fir.select_rank %selector : i32 [1, ^dest(%a, %b : i32, i32),
                                  unit, ^dest(%b, %a : i32, i32)]
^dest(%value: i32, %unused: i32):
  return %value : i32
}

// CHECK-LABEL: func.func @select_type_remove_unused(
// CHECK-SAME: %[[SEL:.*]]: !fir.class<none>, %[[A:.*]]: i32, %[[B:.*]]: i32)
// CHECK: fir.select_type %[[SEL]] : !fir.class<none> [#fir.type_is<i32>, ^[[DEST:bb[0-9]+]](%[[A]] : i32), unit, ^[[DEST]](%[[B]] : i32)]
// CHECK: ^[[DEST]](%[[VALUE:.*]]: i32):
// CHECK-NEXT: return %[[VALUE]] : i32
func.func @select_type_remove_unused(%selector: !fir.class<none>, %a: i32, %b: i32) -> i32 {
  fir.select_type %selector : !fir.class<none> [#fir.type_is<i32>, ^dest(%a, %b : i32, i32),
                                             unit, ^dest(%b, %a : i32, i32)]
^dest(%value: i32, %unused: i32):
  return %value : i32
}

// Merging the two store blocks adds a block argument and an operand to each
// incoming edge. The second block has two incoming edges from select_case;
// both must be updated, while the default edge must remain empty.
// CHECK-LABEL: func.func @select_case_merge_blocks(
// CHECK-SAME: %[[SEL:.*]]: i32, %[[REF:.*]]: !fir.ref<f64>)
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : i32
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : i32
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1.000000e+00 : f64
// CHECK-DAG: %[[TWO:.*]] = arith.constant 2.000000e+00 : f64
// CHECK: fir.select_case %[[SEL]] : i32 [#fir.point, %[[C1]], ^[[STORE:bb[0-9]+]](%[[ONE]] : f64), #fir.point, %[[C2]], ^[[STORE]](%[[TWO]] : f64), #fir.point, %[[C3]], ^[[STORE]](%[[TWO]] : f64), unit, ^[[EXIT:bb[0-9]+]]]
// CHECK: ^[[STORE]](%[[VALUE:.*]]: f64):
// CHECK-NEXT: fir.store %[[VALUE]] to %[[REF]] : !fir.ref<f64>
// CHECK-NEXT: cf.br ^[[EXIT]]
// CHECK: ^[[EXIT]]:
// CHECK-NEXT: return
func.func @select_case_merge_blocks(%selector: i32, %value: !fir.ref<f64>) {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %one = arith.constant 1.0 : f64
  %two = arith.constant 2.0 : f64
  fir.select_case %selector : i32 [#fir.point, %c1, ^first,
                                  #fir.point, %c2, ^second,
                                  #fir.point, %c3, ^second,
                                  unit, ^exit]
^first:
  fir.store %one to %value : !fir.ref<f64>
  cf.br ^exit
^second:
  fir.store %two to %value : !fir.ref<f64>
  cf.br ^exit
^exit:
  return
}
