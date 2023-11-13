! RUN: %flang_fc1 -I nowhere -emit-fir -flang-deprecated-no-hlfir -fopenacc %s -o - | FileCheck %s

! This test checks the lowering of atomic capture

program acc_atomic_capture_test                                                                                                     
    integer :: x, y

!CHECK: %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
!CHECK: %[[Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFEy"}
!CHECK: %[[temp:.*]] = fir.load %[[X]] : !fir.ref<i32>
!CHECK: acc.atomic.capture {
!CHECK: acc.atomic.read %[[X]] = %[[Y]] : !fir.ref<i32>
!CHECK: acc.atomic.update %[[Y]] : !fir.ref<i32> {
!CHECK: ^bb0(%[[ARG:.*]]: i32):
!CHECK: %[[result:.*]] = arith.addi %[[temp]], %[[ARG]] : i32
!CHECK: acc.yield %[[result]] : i32
!CHECK: }
!CHECK: }

    !$acc atomic capture
        x = y
        y = x + y
    !$acc end atomic


!CHECK: %[[temp:.*]] = fir.load %[[X]] : !fir.ref<i32>
!CHECK: acc.atomic.capture {
!CHECK: acc.atomic.update %[[Y]] : !fir.ref<i32> {
!CHECK: ^bb0(%[[ARG:.*]]: i32):
!CHECK: %[[result:.*]] = arith.muli %[[temp]], %[[ARG]] : i32
!CHECK: acc.yield %[[result]] : i32
!CHECK: }
!CHECK: acc.atomic.read %[[X]] = %[[Y]] : !fir.ref<i32>
!CHECK: }

    !$acc atomic capture
        y = x * y 
        x = y
    !$acc end atomic

!CHECK: %[[constant_20:.*]] = arith.constant 20 : i32
!CHECK: %[[constant_8:.*]] = arith.constant 8 : i32
!CHECK: %[[temp:.*]] = fir.load %[[X]] : !fir.ref<i32>
!CHECK: %[[result:.*]] = arith.subi %[[constant_8]], %[[temp]] : i32
!CHECK: %[[result_noreassoc:.*]] = fir.no_reassoc %[[result]] : i32
!CHECK: %[[result:.*]] = arith.addi %[[constant_20]], %[[result_noreassoc]] : i32
!CHECK: acc.atomic.capture {
!CHECK: acc.atomic.read %[[X]] = %[[Y]] : !fir.ref<i32>
!CHECK: acc.atomic.write %[[Y]] = %[[result]] : !fir.ref<i32>, i32
!CHECK: }

    !$acc atomic capture
        x = y
        y = 2 * 10 + (8 - x) 
    !$acc end atomic 
end program



subroutine pointers_in_atomic_capture()
!CHECK: %[[A:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "a", uniq_name = "_QFpointers_in_atomic_captureEa"}
!CHECK: {{.*}} = fir.zero_bits !fir.ptr<i32>
!CHECK: {{.*}} = fir.embox {{.*}} : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
!CHECK: fir.store {{.*}} to %[[A]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[B:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "b", uniq_name = "_QFpointers_in_atomic_captureEb"}
!CHECK: {{.*}} = fir.zero_bits !fir.ptr<i32>
!CHECK: {{.*}} = fir.embox {{.*}} : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
!CHECK: fir.store {{.*}} to %[[B]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[C:.*]] = fir.alloca i32 {bindc_name = "c", fir.target, uniq_name = "_QFpointers_in_atomic_captureEc"}
!CHECK: %[[D:.*]] = fir.alloca i32 {bindc_name = "d", fir.target, uniq_name = "_QFpointers_in_atomic_captureEd"}
!CHECK: {{.*}} = fir.embox {{.*}} : (!fir.ref<i32>) -> !fir.box<!fir.ptr<i32>>
!CHECK: fir.store {{.*}} to %[[A]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: {{.*}} = fir.embox {{.*}} : (!fir.ref<i32>) -> !fir.box<!fir.ptr<i32>>
!CHECK: fir.store {{.*}} to %[[B]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[loaded_A:.*]] = fir.load %[[A]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[loaded_A_addr:.*]] = fir.box_addr %[[loaded_A]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK: %[[loaded_B:.*]] = fir.load %[[B]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[loaded_B_addr:.*]] = fir.box_addr %[[loaded_B]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK: %[[PRIVATE_LOADED_B:.*]] = fir.load %[[B]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[PRIVATE_LOADED_B_addr:.*]] = fir.box_addr %[[PRIVATE_LOADED_B]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK: %[[loaded_value:.*]] = fir.load %[[PRIVATE_LOADED_B_addr]] : !fir.ptr<i32>
!CHECK: acc.atomic.capture   {
!CHECK: acc.atomic.update %[[loaded_A_addr]] : !fir.ptr<i32> {
!CHECK: ^bb0(%[[ARG:.*]]: i32):
!CHECK: %[[result:.*]] = arith.addi %[[ARG]], %[[loaded_value]] : i32
!CHECK: acc.yield %[[result]] : i32
!CHECK: }
!CHECK: acc.atomic.read %[[loaded_B_addr]] = %[[loaded_A_addr]] : !fir.ptr<i32>, i32
!CHECK: }
    integer, pointer :: a, b
    integer, target :: c, d
    a=>c
    b=>d

    !$acc atomic capture
        a = a + b
        b = a
    !$acc end atomic
end subroutine


subroutine capture_with_convert_f32_to_i32()
  implicit none
  integer :: k, v, i

  k = 1
  v = 0

  !$acc atomic capture
  v = k
  k = (i + 1) * 3.14
  !$acc end atomic
end subroutine

! CHECK-LABEL: func.func @_QPcapture_with_convert_f32_to_i32()
! CHECK: %[[K:.*]] = fir.alloca i32 {bindc_name = "k", uniq_name = "_QFcapture_with_convert_f32_to_i32Ek"}
! CHECK: %[[V:.*]] = fir.alloca i32 {bindc_name = "v", uniq_name = "_QFcapture_with_convert_f32_to_i32Ev"}
! CHECK: %[[CST:.*]] = arith.constant 3.140000e+00 : f32
! CHECK: %[[MUL:.*]] = arith.mulf %{{.*}}, %[[CST]] fastmath<contract> : f32
! CHECK: %[[CONV:.*]] = fir.convert %[[MUL]] : (f32) -> i32
! CHECK: acc.atomic.capture {
! CHECK:   acc.atomic.read %[[V]] = %[[K]] : !fir.ref<i32>, i32
! CHECK:   acc.atomic.write %[[K]] = %[[CONV]] : !fir.ref<i32>, i32
! CHECK: }

subroutine array_ref_in_atomic_capture1
  integer :: x(10), v
  !$acc atomic capture
  v = x(7)
  x(7) = x(7) + 1
  !$acc end atomic
end subroutine array_ref_in_atomic_capture1
! CHECK-LABEL:   func.func @_QParray_ref_in_atomic_capture1() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "v", uniq_name = "_QFarray_ref_in_atomic_capture1Ev"}
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.array<10xi32> {bindc_name = "x", uniq_name = "_QFarray_ref_in_atomic_capture1Ex"}
! CHECK:           %[[VAL_5:.*]] = fir.coordinate_of %[[VAL_1]], %{{.*}} : (!fir.ref<!fir.array<10xi32>>, i64) -> !fir.ref<i32>
! CHECK:           acc.atomic.capture {
! CHECK:             acc.atomic.read %[[VAL_0]] = %[[VAL_5]] : !fir.ref<i32>, i32
! CHECK:             acc.atomic.update %[[VAL_5]] : !fir.ref<i32> {
! CHECK:             ^bb0(%[[VAL_7:.*]]: i32):
! CHECK:               %[[VAL_8:.*]] = arith.addi %[[VAL_7]], %{{.*}} : i32
! CHECK:               acc.yield %[[VAL_8]] : i32
! CHECK:             }
! CHECK:           }

subroutine array_ref_in_atomic_capture2
  integer :: x(10), v
  !$acc atomic capture
  x(7) = x(7) + 1
  v = x(7)
  !$acc end atomic
end subroutine array_ref_in_atomic_capture2
! CHECK-LABEL:   func.func @_QParray_ref_in_atomic_capture2() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "v", uniq_name = "_QFarray_ref_in_atomic_capture2Ev"}
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.array<10xi32> {bindc_name = "x", uniq_name = "_QFarray_ref_in_atomic_capture2Ex"}
! CHECK:           %[[VAL_5:.*]] = fir.coordinate_of %[[VAL_1]], %{{.*}} : (!fir.ref<!fir.array<10xi32>>, i64) -> !fir.ref<i32>
! CHECK:           acc.atomic.capture {
! CHECK:             acc.atomic.update %[[VAL_5]] : !fir.ref<i32> {
! CHECK:             ^bb0(%[[VAL_7:.*]]: i32):
! CHECK:               %[[VAL_8:.*]] = arith.addi %[[VAL_7]], %{{.*}} : i32
! CHECK:               acc.yield %[[VAL_8]] : i32
! CHECK:             }
! CHECK:             acc.atomic.read %[[VAL_0]] = %[[VAL_5]] : !fir.ref<i32>, i32
! CHECK:           }

subroutine comp_ref_in_atomic_capture1
  type t1
     integer :: c
  end type t1
  integer :: v
  type(t1) :: x
  !$acc atomic capture
  v = x%c
  x%c = x%c + 1
  !$acc end atomic
end subroutine comp_ref_in_atomic_capture1
! CHECK-LABEL:   func.func @_QPcomp_ref_in_atomic_capture1() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "v", uniq_name = "_QFcomp_ref_in_atomic_capture1Ev"}
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.type<_QFcomp_ref_in_atomic_capture1Tt1{c:i32}> {bindc_name = "x", uniq_name = "_QFcomp_ref_in_atomic_capture1Ex"}
! CHECK:           %[[VAL_2:.*]] = fir.field_index c, !fir.type<_QFcomp_ref_in_atomic_capture1Tt1{c:i32}>
! CHECK:           %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_2]] : (!fir.ref<!fir.type<_QFcomp_ref_in_atomic_capture1Tt1{c:i32}>>, !fir.field) -> !fir.ref<i32>
! CHECK:           acc.atomic.capture {
! CHECK:             acc.atomic.read %[[VAL_0]] = %[[VAL_3]] : !fir.ref<i32>, i32
! CHECK:             acc.atomic.update %[[VAL_3]] : !fir.ref<i32> {
! CHECK:             ^bb0(%[[VAL_5:.*]]: i32):
! CHECK:               %[[VAL_6:.*]] = arith.addi %[[VAL_5]], %{{.*}} : i32
! CHECK:               acc.yield %[[VAL_6]] : i32
! CHECK:             }
! CHECK:           }

subroutine comp_ref_in_atomic_capture2
  type t1
     integer :: c
  end type t1
  integer :: v
  type(t1) :: x
  !$acc atomic capture
  x%c = x%c + 1
  v = x%c
  !$acc end atomic
end subroutine comp_ref_in_atomic_capture2
! CHECK-LABEL:   func.func @_QPcomp_ref_in_atomic_capture2() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "v", uniq_name = "_QFcomp_ref_in_atomic_capture2Ev"}
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.type<_QFcomp_ref_in_atomic_capture2Tt1{c:i32}> {bindc_name = "x", uniq_name = "_QFcomp_ref_in_atomic_capture2Ex"}
! CHECK:           %[[VAL_2:.*]] = fir.field_index c, !fir.type<_QFcomp_ref_in_atomic_capture2Tt1{c:i32}>
! CHECK:           %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_2]] : (!fir.ref<!fir.type<_QFcomp_ref_in_atomic_capture2Tt1{c:i32}>>, !fir.field) -> !fir.ref<i32>
! CHECK:           acc.atomic.capture {
! CHECK:             acc.atomic.update %[[VAL_3]] : !fir.ref<i32> {
! CHECK:             ^bb0(%[[VAL_5:.*]]: i32):
! CHECK:               %[[VAL_6:.*]] = arith.addi %[[VAL_5]], %{{.*}} : i32
! CHECK:               acc.yield %[[VAL_6]] : i32
! CHECK:             }
! CHECK:             acc.atomic.read %[[VAL_0]] = %[[VAL_3]] : !fir.ref<i32>, i32
! CHECK:           }
