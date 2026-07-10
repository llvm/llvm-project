// RUN: fir-opt -flang-licm --split-input-file %s | FileCheck %s

// Test that the box is qualified as present when it is
// passed through arguments of acc.compute_region.
// CHECK-LABEL:   func.func @test_(
// CHECK-SAME:                     %[[ARG0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"},
// CHECK-SAME:                     %[[ARG1:.*]]: index {fir.bindc_name = "n"}) {
// CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[DECLARE_0:.*]] = fir.declare %[[ARG0]] dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {uniq_name = "_QFtestEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
// CHECK:           %[[REBOX_0:.*]] = fir.rebox %[[DECLARE_0]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
// CHECK:           %[[COPYIN_0:.*]] = acc.copyin var(%[[REBOX_0]] : !fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>> {dataClause = #acc<data_clause acc_copy>, implicit = true, name = "x"}
// CHECK:           acc.kernel_environment dataOperands(%[[COPYIN_0]] : !fir.box<!fir.array<?xf32>>) {
// CHECK:             acc.compute_region ins(%[[VAL_0:.*]] = %[[ARG1]], %[[VAL_1:.*]] = %[[COPYIN_0]]) : (index, !fir.box<!fir.array<?xf32>>) {
// CHECK:               %[[BOX_ADDR_0:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
// CHECK:               %[[CONVERT_0:.*]] = fir.convert %[[BOX_ADDR_0]] : (!fir.ref<!fir.array<?xf32>>) -> memref<?xf32>
// CHECK:               %[[BOX_DIMS_0:.*]]:3 = fir.box_dims %[[VAL_1]], %{{.*}} : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
// CHECK:               scf.for %[[VAL_2:.*]] = {{.*}} {
// CHECK:                 %[[BOX_ELESIZE_0:.*]] = fir.box_elesize %[[VAL_1]] : (!fir.box<!fir.array<?xf32>>) -> index
// CHECK:                 %[[DIVSI_0:.*]] = arith.divsi %[[BOX_DIMS_0]]#2, %[[BOX_ELESIZE_0]] : index
// CHECK:                 %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[CONVERT_0]] to offset: [0], sizes: {{\[}}%[[BOX_DIMS_0]]#1], strides: {{\[}}%[[DIVSI_0]]] : memref<?xf32> to memref<?xf32, strided<[?]>>
// CHECK:                 memref.store %{{.*}}, %[[REINTERPRET_CAST_0]]{{\[}}%[[VAL_2]]] : memref<?xf32, strided<[?]>>
// CHECK:               }
// CHECK:               acc.yield
// CHECK:             } {origin = "acc.kernels"}
// CHECK:           }
// CHECK:           acc.copyout accVar(%[[COPYIN_0]] : !fir.box<!fir.array<?xf32>>) to var(%[[REBOX_0]] : !fir.box<!fir.array<?xf32>>) {dataClause = #acc<data_clause acc_copy>, implicit = true, name = "x"}
// CHECK:           return
// CHECK:         }
func.func @test_(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}, %arg1: index {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %4 = fir.declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFtestEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  %5 = fir.rebox %4 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %7 = acc.copyin var(%5 : !fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>> {dataClause = #acc<data_clause acc_copy>, implicit = true, name = "x"}
  acc.kernel_environment dataOperands(%7 : !fir.box<!fir.array<?xf32>>) {
    acc.compute_region ins(%arg2 = %arg1, %arg3 = %7) : (index, !fir.box<!fir.array<?xf32>>) {
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c0 = arith.constant 0 : index
      scf.for %arg4 = %c0 to %arg2 step %c1 {
        %13 = fir.box_addr %arg3 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
        %14 = fir.convert %13 : (!fir.ref<!fir.array<?xf32>>) -> memref<?xf32>
        %17 = fir.box_elesize %arg3 : (!fir.box<!fir.array<?xf32>>) -> index
        %18:3 = fir.box_dims %arg3, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
        %19 = arith.divsi %18#2, %17 : index
        %reinterpret_cast = memref.reinterpret_cast %14 to offset: [0], sizes: [%18#1], strides: [%19] : memref<?xf32> to memref<?xf32, strided<[?]>>
        memref.store %cst, %reinterpret_cast[%arg4] : memref<?xf32, strided<[?]>>
      }
      acc.yield
    } {origin = "acc.kernels"}
  }
  acc.copyout accVar(%7 : !fir.box<!fir.array<?xf32>>) to var(%5 : !fir.box<!fir.array<?xf32>>) {dataClause = #acc<data_clause acc_copy>, implicit = true, name = "x"}
  return
}
