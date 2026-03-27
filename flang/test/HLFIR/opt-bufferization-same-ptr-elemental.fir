// RUN: fir-opt --opt-bufferization %s | FileCheck %s

// Verify that the hlfir.assign of hlfir.elemental is optimized
// into element-per-element assignment:
// subroutine test1(p)
//   real, pointer :: p(:)
//   p = p + 1.0
// end subroutine test1

func.func @_QPtest1(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {fir.bindc_name = "p"}) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 1.000000e+00 : f32
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest1Ep"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
  %2 = fir.load %1#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  %3:3 = fir.box_dims %2, %c0 : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
  %4 = fir.shape %3#1 : (index) -> !fir.shape<1>
  %5 = hlfir.elemental %4 unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
  ^bb0(%arg1: index):
    %6 = arith.subi %3#0, %c1 : index
    %7 = arith.addi %arg1, %6 : index
    %8 = hlfir.designate %2 (%7)  : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> !fir.ref<f32>
    %9 = fir.load %8 : !fir.ref<f32>
    %10 = arith.addf %9, %cst fastmath<contract> : f32
    hlfir.yield_element %10 : f32
  }
  hlfir.assign %5 to %2 : !hlfir.expr<?xf32>, !fir.box<!fir.ptr<!fir.array<?xf32>>>
  hlfir.destroy %5 : !hlfir.expr<?xf32>
  return
}
// CHECK-LABEL:   func.func @_QPtest1(
// CHECK-NOT:         hlfir.assign
// CHECK:             hlfir.assign %{{.*}} to %{{.*}} : f32, !fir.ref<f32>
// CHECK-NOT:         hlfir.assign

// subroutine test2(p)
//   real, pointer :: p(:,:)
//   p = p + 1.0
// end subroutine test2
func.func @_QPtest2(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>> {fir.bindc_name = "p"}) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 1.000000e+00 : f32
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest2Ep"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>)
  %2 = fir.load %1#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
  %3:3 = fir.box_dims %2, %c0 : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index) -> (index, index, index)
  %4:3 = fir.box_dims %2, %c1 : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index) -> (index, index, index)
  %5 = fir.shape %3#1, %4#1 : (index, index) -> !fir.shape<2>
  %6 = hlfir.elemental %5 unordered : (!fir.shape<2>) -> !hlfir.expr<?x?xf32> {
  ^bb0(%arg1: index, %arg2: index):
    %7 = arith.subi %3#0, %c1 : index
    %8 = arith.addi %arg1, %7 : index
    %9 = arith.subi %4#0, %c1 : index
    %10 = arith.addi %arg2, %9 : index
    %11 = hlfir.designate %2 (%8, %10)  : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index, index) -> !fir.ref<f32>
    %12 = fir.load %11 : !fir.ref<f32>
    %13 = arith.addf %12, %cst fastmath<contract> : f32
    hlfir.yield_element %13 : f32
  }
  hlfir.assign %6 to %2 : !hlfir.expr<?x?xf32>, !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
  hlfir.destroy %6 : !hlfir.expr<?x?xf32>
  return
}
// CHECK-LABEL:   func.func @_QPtest2(
// CHECK-NOT:         hlfir.assign
// CHECK:             hlfir.assign %{{.*}} to %{{.*}} : f32, !fir.ref<f32>
// CHECK-NOT:         hlfir.assign
