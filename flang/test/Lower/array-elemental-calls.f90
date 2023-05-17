! Test lowering of elemental calls in array expressions.
! RUN: bbc -o - -emit-fir %s | FileCheck %s

module scalar_in_elem

contains
elemental integer function elem_by_ref(a,b) result(r)
  integer, intent(in) :: a
  real, intent(in) :: b
  r = a + b
end function
elemental integer function elem_by_valueref(a,b) result(r)
  integer, value :: a
  real, value :: b
  r = a + b
end function

! CHECK-LABEL: func @_QMscalar_in_elemPtest_elem_by_ref(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.array<100xi32>>{{.*}}, %[[arg1:.*]]: !fir.ref<!fir.array<100xi32>>{{.*}}) {
subroutine test_elem_by_ref(i, j)
  integer :: i(100), j(100)
  ! CHECK: %[[tmp:.*]] = fir.alloca f32
  ! CHECK: %[[cst:.*]] = arith.constant 4.200000e+01 : f32
  ! CHECK: fir.store %[[cst]] to %[[tmp]] : !fir.ref<f32>

  ! CHECK: fir.do_loop
    ! CHECK: %[[j:.*]] = fir.array_coor %[[arg1]](%{{.*}}) %{{.*}} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    ! CHECK: fir.call @_QMscalar_in_elemPelem_by_ref(%[[j]], %[[tmp]]) {{.*}}: (!fir.ref<i32>, !fir.ref<f32>) -> i32
    ! CHECK: fir.result
  i = elem_by_ref(j, 42.)
end

! CHECK-LABEL: func @_QMscalar_in_elemPtest_elem_by_valueref(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.array<100xi32>>{{.*}}, %[[arg1:.*]]: !fir.ref<!fir.array<100xi32>>{{.*}}) {
subroutine test_elem_by_valueref(i, j)
  integer :: i(100), j(100)
  ! CHECK: %[[jload:.*]] = fir.array_load %[[arg1]]
  ! CHECK: %[[cst:.*]] = arith.constant 4.200000e+01 : f32

  ! CHECK: fir.do_loop
    ! CHECK: %[[j:.*]] = fir.array_fetch %[[jload]], %{{.*}} : (!fir.array<100xi32>, index) -> i32
    ! CHECK: fir.call @_QMscalar_in_elemPelem_by_valueref(%[[j]], %[[cst]]) {{.*}}: (i32, f32) -> i32
    ! CHECK: fir.result
  i = elem_by_valueref(j, 42.)
end
end module


! Test that impure elemental functions cause ordered loops to be emitted
subroutine test_loop_order(i, j)
  integer :: i(:), j(:)
  interface
    elemental integer function pure_func(j)
      integer, intent(in) :: j
    end function
    elemental impure integer function impure_func(j)
      integer, intent(in) :: j
    end function
  end interface
  
  i = 42 + pure_func(j)
  i = 42 + impure_func(j)
end subroutine

! CHECK-LABEL: func @_QPtest_loop_order(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}, %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}) {
! CHECK:         %[[VAL_2:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_3:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_2]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:         %[[VAL_4:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
! CHECK:         %[[VAL_5:.*]] = arith.constant 42 : i32
! CHECK:         %[[VAL_6:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_7:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_8:.*]] = arith.subi %[[VAL_3]]#1, %[[VAL_6]] : index
! CHECK:         %[[VAL_9:.*]] = fir.do_loop %[[VAL_10:.*]] = %[[VAL_7]] to %[[VAL_8]] step %[[VAL_6]] unordered iter_args(%[[VAL_11:.*]] = %[[VAL_4]]) -> (!fir.array<?xi32>) {
! CHECK:           %[[VAL_12:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_13:.*]] = arith.addi %[[VAL_10]], %[[VAL_12]] : index
! CHECK:           %[[VAL_14:.*]] = fir.array_coor %[[VAL_1]] %[[VAL_13]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:           %[[VAL_15:.*]] = fir.call @_QPpure_func(%[[VAL_14]]) {{.*}}: (!fir.ref<i32>) -> i32
! CHECK:           %[[VAL_16:.*]] = arith.addi %[[VAL_5]], %[[VAL_15]] : i32
! CHECK:           %[[VAL_17:.*]] = fir.array_update %[[VAL_11]], %[[VAL_16]], %[[VAL_10]] : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
! CHECK:           fir.result %[[VAL_17]] : !fir.array<?xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_4]], %[[VAL_18:.*]] to %[[VAL_0]] : !fir.array<?xi32>, !fir.array<?xi32>, !fir.box<!fir.array<?xi32>>
! CHECK:         %[[VAL_19:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_20:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_19]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:         %[[VAL_21:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
! CHECK:         %[[VAL_22:.*]] = arith.constant 42 : i32
! CHECK:         %[[VAL_23:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_24:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_25:.*]] = arith.subi %[[VAL_20]]#1, %[[VAL_23]] : index
! CHECK:         %[[VAL_26:.*]] = fir.do_loop %[[VAL_27:.*]] = %[[VAL_24]] to %[[VAL_25]] step %[[VAL_23]] iter_args(%[[VAL_28:.*]] = %[[VAL_21]]) -> (!fir.array<?xi32>) {
! CHECK:           %[[VAL_29:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_30:.*]] = arith.addi %[[VAL_27]], %[[VAL_29]] : index
! CHECK:           %[[VAL_31:.*]] = fir.array_coor %[[VAL_1]] %[[VAL_30]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:           %[[VAL_32:.*]] = fir.call @_QPimpure_func(%[[VAL_31]]) {{.*}}: (!fir.ref<i32>) -> i32
! CHECK:           %[[VAL_33:.*]] = arith.addi %[[VAL_22]], %[[VAL_32]] : i32
! CHECK:           %[[VAL_34:.*]] = fir.array_update %[[VAL_28]], %[[VAL_33]], %[[VAL_27]] : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
! CHECK:           fir.result %[[VAL_34]] : !fir.array<?xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_21]], %[[VAL_35:.*]] to %[[VAL_0]] : !fir.array<?xi32>, !fir.array<?xi32>, !fir.box<!fir.array<?xi32>>
! CHECK:         return
! CHECK:       }
