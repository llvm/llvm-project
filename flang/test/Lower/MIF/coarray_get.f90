
! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPtest_coarray_get_scalar
subroutine test_coarray_get_scalar
  real, save :: a[*]
  real :: b
  b = 2
  b = a[2]
  ! CHECK: %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[VAL_1:.*]] = fir.address_of(@_QFtest_coarray_get_scalarEa) : !fir.ref<!fir.box<!fir.heap<f32>, corank:1>>
  ! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFtest_coarray_get_scalarEa"} : (!fir.ref<!fir.box<!fir.heap<f32>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<f32>, corank:1>>, !fir.ref<!fir.box<!fir.heap<f32>, corank:1>>)
  ! CHECK: %[[VAL_3:.*]] = fir.alloca f32 {bindc_name = "b", uniq_name = "_QFtest_coarray_get_scalarEb"}
  ! CHECK: %[[VAL_4:.*]]:2 = hlfir.declare %3 {uniq_name = "_QFtest_coarray_get_scalarEb"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  ! CHECK: %cst = arith.constant 2.000000e+00 : f32
  ! CHECK: hlfir.assign %cst to %4#0 : f32, !fir.ref<f32>
  ! CHECK: %[[VAL_5:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<f32>, corank:1>>
  ! CHECK: %[[VAL_6:.*]] = fir.box_addr %[[VAL_5]] : (!fir.box<!fir.heap<f32>, corank:1>) -> !fir.heap<f32>
  ! CHECK: %[[VAL_7:.*]] = hlfir.designate %[[VAL_6]]   : (!fir.heap<f32>) -> !fir.ref<f32>
  ! CHECK: %c2_i64 = arith.constant 2 : i64
  ! CHECK: mif.get_coarray from %[[VAL_7]][%c2_i64] to %[[VAL_4]]#0 : (!fir.ref<f32>, i64, !fir.ref<f32>) -> ()
end subroutine

! CHECK-LABEL: func.func @_QPtest_coarray_get_array
subroutine test_coarray_get_array
  integer :: me
  integer, save :: a(3,4)[*], b(3,4)
  
  me = this_image()
  if (me == 1) then
    a = reshape([4, 2, 7, 1, 2, 4, 5, 23, 25, 78, 54, 63], [3,4])
  else
    b = a(:,:)[1]
  endif
  ! CHECK: %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[VAL_1:.*]] = fir.address_of(@_QFtest_coarray_get_arrayEa) : !fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>
  ! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFtest_coarray_get_arrayEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>, !fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>)
  ! CHECK: %[[VAL_3:.*]] = fir.address_of(@_QFtest_coarray_get_arrayEb) : !fir.ref<!fir.array<3x4xi32>>
  ! CHECK: %c3 = arith.constant 3 : index
  ! CHECK: %c4 = arith.constant 4 : index
  ! CHECK: %[[VAL_4:.*]] = fir.shape %c3, %c4 : (index, index) -> !fir.shape<2>
  ! CHECK: %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_3]](%[[VAL_4]]) {uniq_name = "_QFtest_coarray_get_arrayEb"} : (!fir.ref<!fir.array<3x4xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<3x4xi32>>, !fir.ref<!fir.array<3x4xi32>>)
  ! CHECK: %[[VAL_6:.*]] = fir.alloca i32 {bindc_name = "me", uniq_name = "_QFtest_coarray_get_arrayEme"}
  ! CHECK: %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] {uniq_name = "_QFtest_coarray_get_arrayEme"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK: %[[VAL_8:.*]] = mif.this_image : () -> i32
  ! CHECK: hlfir.assign %[[VAL_8]] to %[[VAL_7]]#0 : i32, !fir.ref<i32>
  ! CHECK: %[[VAL_9:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<i32>
  ! CHECK: %c1_i32 = arith.constant 1 : i32
  ! CHECK: %[[VAL_10:.*]] = arith.cmpi eq, %[[VAL_9]], %c1_i32 : i32
  ! CHECK: fir.if %[[VAL_10]] {
  ! CHECK:   %[[VAL_11:.*]] = fir.address_of(@_QQro.3x4xi4.0) : !fir.ref<!fir.array<3x4xi32>>
  ! CHECK:   %c3_0 = arith.constant 3 : index
  ! CHECK:   %c4_1 = arith.constant 4 : index
  ! CHECK:   %[[VAL_12:.*]] = fir.shape %c3_0, %c4_1 : (index, index) -> !fir.shape<2>
  ! CHECK:   %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_11]](%[[VAL_12]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.3x4xi4.0"} : (!fir.ref<!fir.array<3x4xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<3x4xi32>>, !fir.ref<!fir.array<3x4xi32>>)
  ! CHECK:   mif.put_coarray from %[[VAL_13]]#0 to %[[VAL_2]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>, !fir.ref<!fir.array<3x4xi32>>) -> ()
  ! CHECK: } else {
  ! CHECK:   %[[VAL_11:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>
  ! CHECK:   %c0 = arith.constant 0 : index
  ! CHECK:   %[[VAL_12:.*]]:3 = fir.box_dims %[[VAL_11]], %c0 : (!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>, index) -> (index, index, index)
  ! CHECK:   %c1 = arith.constant 1 : index
  ! CHECK:   %[[VAL_13:.*]]:3 = fir.box_dims %[[VAL_11]], %c1 : (!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>, index) -> (index, index, index)
  ! CHECK:   %c1_0 = arith.constant 1 : index
  ! CHECK:   %c0_1 = arith.constant 0 : index
  ! CHECK:   %[[VAL_14:.*]]:3 = fir.box_dims %[[VAL_11]], %c0_1 : (!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>, index) -> (index, index, index)
  ! CHECK:   %[[VAL_15:.*]] = arith.addi %[[VAL_12]]#0, %[[VAL_14]]#1 : index
  ! CHECK:   %[[VAL_16:.*]] = arith.subi %[[VAL_15]], %c1_0 : index
  ! CHECK:   %c1_2 = arith.constant 1 : index
  ! CHECK:   %[[VAL_17:.*]]:3 = fir.box_dims %[[VAL_11]], %c1_2 : (!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>, index) -> (index, index, index)
  ! CHECK:   %[[VAL_18:.*]] = arith.addi %[[VAL_13]]#0, %[[VAL_17]]#1 : index
  ! CHECK:   %[[VAL_19:.*]] = arith.subi %[[VAL_18]], %c1_0 : index
  ! CHECK:   %c1_3 = arith.constant 1 : index
  ! CHECK:   %c3_4 = arith.constant 3 : index
  ! CHECK:   %c1_5 = arith.constant 1 : index
  ! CHECK:   %c4_6 = arith.constant 4 : index
  ! CHECK:   %[[VAL_20:.*]] = fir.shape %c3_4, %c4_6 : (index, index) -> !fir.shape<2>
  ! CHECK:   %[[VAL_21:.*]] = hlfir.designate %[[VAL_11]] (%[[VAL_12]]#0:%[[VAL_16]]:%c1_3, %[[VAL_13]]#0:%[[VAL_19]]:%c1_5)  shape %20 : (!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.ref<!fir.array<3x4xi32>>
  ! CHECK:   %c1_i64 = arith.constant 1 : i64
  ! CHECK:   mif.get_coarray from %[[VAL_21]][%c1_i64] to %[[VAL_5]]#0 : (!fir.ref<!fir.array<3x4xi32>>, i64, !fir.ref<!fir.array<3x4xi32>>) -> ()
  ! CHECK: }
end subroutine

