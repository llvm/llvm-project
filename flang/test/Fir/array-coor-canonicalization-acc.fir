// RUN: fir-opt --canonicalize --split-input-file %s | FileCheck %s

// Verify that fir.rebox is NOT folded into fir.array_coor when the rebox
// result is used by an acc.copyin. The data legalization pipeline depends on
// the copyin's var (the rebox result) matching the value used inside compute
// regions; folding would break this chain and produce an illegal live-in.
// CHECK-LABEL:   func.func @test_acc_copyin_preserves_rebox(
// CHECK-SAME:        %[[BOX:.*]]: !fir.box<!fir.array<?xf32>>,
// CHECK-SAME:        %[[VAL:.*]]: f32) -> !fir.ref<f32> {
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[REBOX:.*]] = fir.rebox %[[BOX]]
// CHECK:           %[[COPYIN:.*]] = acc.copyin var(%[[REBOX]]
// CHECK:           acc.parallel dataOperands(%[[COPYIN]] :
// CHECK:             %[[COOR:.*]] = fir.array_coor %[[REBOX]] %[[C1]]
// CHECK:             fir.store %[[VAL]] to %[[COOR]]
// CHECK:             acc.yield
// CHECK:           }
// CHECK:           acc.copyout accVar(%[[COPYIN]]
// CHECK:           %[[COOR_OUT:.*]] = fir.array_coor %[[BOX]] %[[C1]]
// CHECK:           return %[[COOR_OUT]] : !fir.ref<f32>
func.func @test_acc_copyin_preserves_rebox(%arg0: !fir.box<!fir.array<?xf32>>, %arg1: f32) -> !fir.ref<f32> {
  %c1 = arith.constant 1 : index
  %0 = fir.rebox %arg0 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %1 = acc.copyin var(%0 : !fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>> {dataClause = #acc<data_clause acc_copy>, name = "array"}
  acc.parallel dataOperands(%1 : !fir.box<!fir.array<?xf32>>) {
    %2 = fir.array_coor %0 %c1 : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
    fir.store %arg1 to %2 : !fir.ref<f32>
    acc.yield
  } attributes {defaultAttr = #acc<defaultvalue none>}
  acc.copyout accPtr(%1 : !fir.box<!fir.array<?xf32>>) to varPtr(%0 : !fir.box<!fir.array<?xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "array"}
  %3 = fir.array_coor %0 %c1 : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
  return %3 : !fir.ref<f32>
}

// -----

// Same scenario but without acc.copyin: rebox SHOULD be folded normally.
// CHECK-LABEL:   func.func @test_no_acc_folds_rebox(
// CHECK-SAME:        %[[BOX:.*]]: !fir.box<!fir.array<?xf32>>) -> !fir.ref<f32> {
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[COOR:.*]] = fir.array_coor %[[BOX]] %[[C1]]
// CHECK:           return %[[COOR]] : !fir.ref<f32>
func.func @test_no_acc_folds_rebox(%arg0: !fir.box<!fir.array<?xf32>>) -> !fir.ref<f32> {
  %c1 = arith.constant 1 : index
  %0 = fir.rebox %arg0 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %2 = fir.array_coor %0 %c1 : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
  return %2 : !fir.ref<f32>
}
