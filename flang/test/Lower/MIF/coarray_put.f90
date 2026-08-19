! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPtest_coarray_put_scalar
subroutine test_coarray_put_scalar
  real, save :: a[*]
  a = 2
  a[2] = 3
  ! CHECK: %[[VAL_0:.*]] = fir.alloca f32
  ! CHECK: %[[VAL_1:.*]] = fir.alloca f32
  ! CHECK: %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[VAL_3:.*]] = fir.address_of(@_QFtest_coarray_put_scalarEa) : !fir.ref<!fir.box<!fir.heap<f32>, corank:1>>
  ! CHECK: %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = "_QFtest_coarray_put_scalarEa"} : (!fir.ref<!fir.box<!fir.heap<f32>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<f32>, corank:1>>, !fir.ref<!fir.box<!fir.heap<f32>, corank:1>>)
  ! CHECK: %cst = arith.constant 2.000000e+00 : f32
  ! CHECK: fir.store %cst to %[[VAL_1]] : !fir.ref<f32>
  ! CHECK: mif.put_coarray from %[[VAL_1]] to %[[VAL_4]]#0 : (!fir.ref<!fir.box<!fir.heap<f32>, corank:1>>, !fir.ref<f32>) -> ()
  ! CHECK: %[[VAL_5:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<!fir.box<!fir.heap<f32>, corank:1>>
  ! CHECK: %[[VAL_6:.*]] = fir.box_addr %[[VAL_5]] : (!fir.box<!fir.heap<f32>, corank:1>) -> !fir.heap<f32>
  ! CHECK: %[[VAL_7:.*]] = hlfir.designate %[[VAL_6]]   : (!fir.heap<f32>) -> !fir.ref<f32>
  ! CHECK: %cst_0 = arith.constant 3.000000e+00 : f32
  ! CHECK: fir.store %cst_0 to %[[VAL_0]] : !fir.ref<f32>
  ! CHECK: %c2_i64 = arith.constant 2 : i64
  ! CHECK: mif.put_coarray from %[[VAL_0]] to %[[VAL_7]][%c2_i64] : (!fir.ref<f32>, i64, !fir.ref<f32>) -> ()
end subroutine

! CHECK-LABEL: func.func @_QPtest_coarray_put_array
subroutine test_coarray_put_array
  integer :: me
  integer, save :: a(3,4)[*]
  
  me = this_image()
  if (me == 1) then
    a = reshape([4, 2, 7, 1, 2, 4, 5, 23, 25, 78, 54, 63], [3,4])
  else
    a(2,3)[2] = 2
  endif
  ! CHECK: %[[VAL_0:.*]] = fir.alloca i32
  ! CHECK: %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[VAL_2:.*]] = fir.address_of(@_QFtest_coarray_put_arrayEa) : !fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>
  ! CHECK: %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFtest_coarray_put_arrayEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>, !fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>)
  ! CHECK: %[[VAL_4:.*]] = fir.alloca i32 {bindc_name = "me", uniq_name = "_QFtest_coarray_put_arrayEme"}
  ! CHECK: %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {uniq_name = "_QFtest_coarray_put_arrayEme"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK: %[[VAL_6:.*]] = mif.this_image : () -> i32
  ! CHECK: hlfir.assign %[[VAL_6]] to %[[VAL_5]]#0 : i32, !fir.ref<i32>
  ! CHECK: %[[VAL_7:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
  ! CHECK: %c1_i32 = arith.constant 1 : i32
  ! CHECK: %[[VAL_8:.*]] = arith.cmpi eq, %[[VAL_7]], %c1_i32 : i32
  ! CHECK: fir.if %[[VAL_8]] {
  ! CHECK:   %[[VAL_9:.*]] = fir.address_of(@_QQro.3x4xi4.0) : !fir.ref<!fir.array<3x4xi32>>
  ! CHECK:   %c3 = arith.constant 3 : index
  ! CHECK:   %c4 = arith.constant 4 : index
  ! CHECK:   %[[VAL_10:.*]] = fir.shape %c3, %c4 : (index, index) -> !fir.shape<2>
  ! CHECK:   %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_9]](%[[VAL_10]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.3x4xi4.0"} : (!fir.ref<!fir.array<3x4xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<3x4xi32>>, !fir.ref<!fir.array<3x4xi32>>)
  ! CHECK:   mif.put_coarray from %[[VAL_11]]#0 to %[[VAL_3]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>, !fir.ref<!fir.array<3x4xi32>>) -> ()
  ! CHECK: } else {
  ! CHECK:   %[[VAL_9:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>
  ! CHECK:   %c2 = arith.constant 2 : index
  ! CHECK:   %c3 = arith.constant 3 : index
  ! CHECK:   %[[VAL_10:.*]] = hlfir.designate %[[VAL_9]] (%c2, %c3)  : (!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>, index, index) -> !fir.ref<i32>
  ! CHECK:   %c2_i32 = arith.constant 2 : i32
  ! CHECK:   fir.store %c2_i32 to %[[VAL_0]] : !fir.ref<i32>
  ! CHECK:   %c2_i64 = arith.constant 2 : i64
  ! CHECK:   mif.put_coarray from %[[VAL_0]] to %[[VAL_10]][%c2_i64] : (!fir.ref<i32>, i64, !fir.ref<i32>) -> ()
  ! CHECK: }

end subroutine
