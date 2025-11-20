// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' 2>&1 | FileCheck %s

// CHECK: test_y(1)#0 <-> test_x(1)#0: MayAlias
func.func @_QPtest(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {fir.bindc_name = "y"}) {
  %c1 = arith.constant 1 : index
  %0 = fir.address_of(@_QMdataEx) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %1 = fir.declare %0 {fortran_attrs = #fir.var_attrs<allocatable, target>, uniq_name = "_QMdataEx"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %2 = fir.declare %arg0 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtestEy"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  %3 = fir.load %2 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  %c0 = arith.constant 0 : index
  %4:3 = fir.box_dims %3, %c0 : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
  %5 = fir.shift %4#0 : (index) -> !fir.shift<1>
  %6 = fir.array_coor %3(%5) %c1 {test.ptr = "test_y(1)"} : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.shift<1>, index) -> !fir.ref<f32>
  %7 = fir.load %6 : !fir.ref<f32>
  %8 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %9 = fir.box_addr %8 : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  %10:3 = fir.box_dims %8, %c0 : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
  %11 = fir.shape_shift %10#0, %10#1 : (index, index) -> !fir.shapeshift<1>
  %12 = fir.array_coor %9(%11) %c1 {test.ptr = "test_x(1)"} : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>, index) -> !fir.ref<f32>
  fir.store %7 to %12 : !fir.ref<f32>
  return
}
fir.global @_QMdataEx target : !fir.box<!fir.heap<!fir.array<?xf32>>> 

