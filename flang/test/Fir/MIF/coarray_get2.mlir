// RUN: not fir-opt --mif-convert %s 2>&1 | FileCheck %s

// CHECK: not yet implemented: coarray: get operation with strides

func.func @_QPtest_coarray_get_array() {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@_QFtest_coarray_get_arrayEa) : !fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>
  %2:2 = hlfir.declare %1 {uniq_name = "_QFtest_coarray_get_arrayEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>, !fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>)
  %3 = fir.address_of(@_QFtest_coarray_get_arrayEb) : !fir.ref<!fir.array<3x4xi32>>
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %4 = fir.shape %c3, %c4 : (index, index) -> !fir.shape<2>
  %5:2 = hlfir.declare %3(%4) {uniq_name = "_QFtest_coarray_get_arrayEb"} : (!fir.ref<!fir.array<3x4xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<3x4xi32>>, !fir.ref<!fir.array<3x4xi32>>)
  %6 = fir.alloca i32 {bindc_name = "me", uniq_name = "_QFtest_coarray_get_arrayEme"}
  %7:2 = hlfir.declare %6 {uniq_name = "_QFtest_coarray_get_arrayEme"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %8 = mif.this_image : () -> i32
  hlfir.assign %8 to %7#0 : i32, !fir.ref<i32>
  %9 = fir.load %7#0 : !fir.ref<i32>
  %c1_i32 = arith.constant 1 : i32
  %10 = arith.cmpi eq, %9, %c1_i32 : i32
  fir.if %10 {
    %11 = fir.address_of(@_QQro.3x4xi4.0) : !fir.ref<!fir.array<3x4xi32>>
    %c3_0 = arith.constant 3 : index
    %c4_1 = arith.constant 4 : index
    %12 = fir.shape %c3_0, %c4_1 : (index, index) -> !fir.shape<2>
    %13:2 = hlfir.declare %11(%12) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.3x4xi4.0"} : (!fir.ref<!fir.array<3x4xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<3x4xi32>>, !fir.ref<!fir.array<3x4xi32>>)
    mif.put_coarray from %13#0 to %2#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>, !fir.ref<!fir.array<3x4xi32>>) -> ()
  } else {
    %11 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>
    %c0 = arith.constant 0 : index
    %12:3 = fir.box_dims %11, %c0 : (!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>, index) -> (index, index, index)
    %c1 = arith.constant 1 : index
    %13:3 = fir.box_dims %11, %c1 : (!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>, index) -> (index, index, index)
    %c1_0 = arith.constant 1 : index
    %c0_1 = arith.constant 0 : index
    %14:3 = fir.box_dims %11, %c0_1 : (!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>, index) -> (index, index, index)
    %15 = arith.addi %12#0, %14#1 : index
    %16 = arith.subi %15, %c1_0 : index
    %c1_2 = arith.constant 1 : index
    %17:3 = fir.box_dims %11, %c1_2 : (!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>, index) -> (index, index, index)
    %18 = arith.addi %13#0, %17#1 : index
    %19 = arith.subi %18, %c1_0 : index
    %c1_3 = arith.constant 1 : index
    %c3_4 = arith.constant 3 : index
    %c1_5 = arith.constant 1 : index
    %c4_6 = arith.constant 4 : index
    %20 = fir.shape %c3_4, %c4_6 : (index, index) -> !fir.shape<2>
    %21 = hlfir.designate %11 (%12#0:%16:%c1_3, %13#0:%19:%c1_5)  shape %20 : (!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.ref<!fir.array<3x4xi32>>
    %c1_i64 = arith.constant 1 : i64
    mif.get_coarray from %21[%c1_i64] to %5#0 : (!fir.ref<!fir.array<3x4xi32>>, i64, !fir.ref<!fir.array<3x4xi32>>) -> ()
  }
  return
}
