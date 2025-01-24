! RUN: %flang_fc1 -I nowhere -emit-hlfir -fopenacc %s -o - | FileCheck %s

! This test checks the lowering of atomic capture

program acc_atomic_capture_test
    integer :: x, y

!CHECK: %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
!CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %0 {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFEy"}
!CHECK: %[[Y_DECL:.*]]:2 = hlfir.declare %2 {uniq_name = "_QFEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[temp:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<i32>
!CHECK: acc.atomic.capture {
!CHECK: acc.atomic.read %[[X_DECL]]#1 = %[[Y_DECL]]#1 : !fir.ref<i32>, !fir.ref<i32>, i32
!CHECK: acc.atomic.update %[[Y_DECL]]#1 : !fir.ref<i32> {
!CHECK: ^bb0(%[[ARG:.*]]: i32):
!CHECK: %[[result:.*]] = arith.addi %[[temp]], %[[ARG]] : i32
!CHECK: acc.yield %[[result]] : i32
!CHECK: }
!CHECK: }

    !$acc atomic capture
        x = y
        y = x + y
    !$acc end atomic


!CHECK: %[[temp:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<i32>
!CHECK: acc.atomic.capture {
!CHECK: acc.atomic.update %[[Y_DECL]]#1 : !fir.ref<i32> {
!CHECK: ^bb0(%[[ARG:.*]]: i32):
!CHECK: %[[result:.*]] = arith.muli %[[temp]], %[[ARG]] : i32
!CHECK: acc.yield %[[result]] : i32
!CHECK: }
!CHECK: acc.atomic.read %[[X_DECL]]#1 = %[[Y_DECL]]#1 : !fir.ref<i32>, !fir.ref<i32>, i32
!CHECK: }

    !$acc atomic capture
        y = x * y 
        x = y
    !$acc end atomic

!CHECK: %[[constant_20:.*]] = arith.constant 20 : i32
!CHECK: %[[constant_8:.*]] = arith.constant 8 : i32
!CHECK: %[[temp:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<i32>
!CHECK: %[[result:.*]] = arith.subi %[[constant_8]], %[[temp]] : i32
!CHECK: %[[result_noreassoc:.*]] = hlfir.no_reassoc %[[result]] : i32
!CHECK: %[[result:.*]] = arith.addi %[[constant_20]], %[[result_noreassoc]] : i32
!CHECK: acc.atomic.capture {
!CHECK: acc.atomic.read %[[X_DECL]]#1 = %[[Y_DECL]]#1 : !fir.ref<i32>, !fir.ref<i32>, i32
!CHECK: acc.atomic.write %[[Y_DECL]]#1 = %[[result]] : !fir.ref<i32>, i32
!CHECK: }

    !$acc atomic capture
        x = y
        y = 2 * 10 + (8 - x) 
    !$acc end atomic 
end program



subroutine pointers_in_atomic_capture()
!CHECK: %[[A:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "a", uniq_name = "_QFpointers_in_atomic_captureEa"}
!CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[A]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFpointers_in_atomic_captureEa"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK: %[[B:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "b", uniq_name = "_QFpointers_in_atomic_captureEb"}
!CHECK: %[[B_DECL:.*]]:2 = hlfir.declare %[[B]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFpointers_in_atomic_captureEb"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK: %[[C:.*]] = fir.alloca i32 {bindc_name = "c", fir.target, uniq_name = "_QFpointers_in_atomic_captureEc"}
!CHECK: %[[C_DECL:.*]]:2 = hlfir.declare %[[C]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFpointers_in_atomic_captureEc"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[D:.*]] = fir.alloca i32 {bindc_name = "d", fir.target, uniq_name = "_QFpointers_in_atomic_captureEd"}
!CHECK: %[[D_DECL:.*]]:2 = hlfir.declare %[[D]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFpointers_in_atomic_captureEd"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

!CHECK: %[[loaded_A:.*]] = fir.load %[[A_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[loaded_A_addr:.*]] = fir.box_addr %[[loaded_A]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK: %[[loaded_B:.*]] = fir.load %[[B_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[loaded_B_addr:.*]] = fir.box_addr %[[loaded_B]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK: %[[PRIVATE_LOADED_B:.*]] = fir.load %[[B_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[PRIVATE_LOADED_B_addr:.*]] = fir.box_addr %[[PRIVATE_LOADED_B]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK: %[[loaded_value:.*]] = fir.load %[[PRIVATE_LOADED_B_addr]] : !fir.ptr<i32>
!CHECK: acc.atomic.capture   {
!CHECK: acc.atomic.update %[[loaded_A_addr]] : !fir.ptr<i32> {
!CHECK: ^bb0(%[[ARG:.*]]: i32):
!CHECK: %[[result:.*]] = arith.addi %[[ARG]], %[[loaded_value]] : i32
!CHECK: acc.yield %[[result]] : i32
!CHECK: }
!CHECK: acc.atomic.read %[[loaded_B_addr]] = %[[loaded_A_addr]] : !fir.ptr<i32>, !fir.ptr<i32>, i32
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
! CHECK: %[[K_DECL:.*]]:2 = hlfir.declare %[[K]] {uniq_name = "_QFcapture_with_convert_f32_to_i32Ek"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[V:.*]] = fir.alloca i32 {bindc_name = "v", uniq_name = "_QFcapture_with_convert_f32_to_i32Ev"}
! CHECK: %[[V_DECL:.*]]:2 = hlfir.declare %[[V]] {uniq_name = "_QFcapture_with_convert_f32_to_i32Ev"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[CST:.*]] = arith.constant 3.140000e+00 : f32
! CHECK: %[[MUL:.*]] = arith.mulf %{{.*}}, %[[CST]] fastmath<contract> : f32
! CHECK: %[[CONV:.*]] = fir.convert %[[MUL]] : (f32) -> i32
! CHECK: acc.atomic.capture {
! CHECK:   acc.atomic.read %[[V_DECL]]#1 = %[[K_DECL]]#1 : !fir.ref<i32>, !fir.ref<i32>, i32
! CHECK:   acc.atomic.write %[[K_DECL]]#1 = %[[CONV]] : !fir.ref<i32>, i32
! CHECK: }

subroutine capture_with_convert_i32_to_f64()
  real(8) :: x
  integer :: v
  x = 1.0
  v = 0
  !$acc atomic capture
  v = x
  x = v
  !$acc end atomic
end subroutine capture_with_convert_i32_to_f64

! CHECK-LABEL: func.func @_QPcapture_with_convert_i32_to_f64()
! CHECK: %[[V:.*]] = fir.alloca i32 {bindc_name = "v", uniq_name = "_QFcapture_with_convert_i32_to_f64Ev"}
! CHECK: %[[V_DECL:.*]]:2 = hlfir.declare %[[V]] {uniq_name = "_QFcapture_with_convert_i32_to_f64Ev"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[X:.*]] = fir.alloca f64 {bindc_name = "x", uniq_name = "_QFcapture_with_convert_i32_to_f64Ex"}
! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFcapture_with_convert_i32_to_f64Ex"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
! CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f64
! CHECK: hlfir.assign %[[CST]] to %[[X_DECL]]#0 : f64, !fir.ref<f64>
! CHECK: %c0_i32 = arith.constant 0 : i32
! CHECK: hlfir.assign %c0_i32 to %[[V_DECL]]#0 : i32, !fir.ref<i32>
! CHECK: %[[LOAD:.*]] = fir.load %[[V_DECL]]#0 : !fir.ref<i32>
! CHECK: %[[CONV:.*]] = fir.convert %[[LOAD]] : (i32) -> f64
! CHECK: acc.atomic.capture {
! CHECK:   acc.atomic.read %[[V_DECL]]#1 = %[[X_DECL]]#1 : !fir.ref<i32>, !fir.ref<f64>, f64
! CHECK:   acc.atomic.write %[[X_DECL]]#1 = %[[CONV]] : !fir.ref<f64>, f64
! CHECK: }

subroutine capture_with_convert_f64_to_i32()
  integer :: x
  real(8) :: v
  x = 1
  v = 0
  !$acc atomic capture
  x = v * v
  v = x
  !$acc end atomic
end subroutine capture_with_convert_f64_to_i32

! CHECK-LABEL: func.func @_QPcapture_with_convert_f64_to_i32()
! CHECK: %[[V:.*]] = fir.alloca f64 {bindc_name = "v", uniq_name = "_QFcapture_with_convert_f64_to_i32Ev"}
! CHECK: %[[V_DECL:.*]]:2 = hlfir.declare %[[V]] {uniq_name = "_QFcapture_with_convert_f64_to_i32Ev"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
! CHECK: %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFcapture_with_convert_f64_to_i32Ex"}
! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFcapture_with_convert_f64_to_i32Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %c1_i32 = arith.constant 1 : i32
! CHECK: hlfir.assign %c1_i32 to %[[X_DECL]]#0 : i32, !fir.ref<i32>
! CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f64
! CHECK: hlfir.assign %[[CST]] to %[[V_DECL]]#0 : f64, !fir.ref<f64>
! CHECK: %[[LOAD:.*]] = fir.load %[[V_DECL]]#0 : !fir.ref<f64>
! CHECK: acc.atomic.capture {
! CHECK:   acc.atomic.update %[[X_DECL]]#1 : !fir.ref<i32> {
! CHECK:   ^bb0(%arg0: i32):
! CHECK:     %[[MUL:.*]] = arith.mulf %[[LOAD]], %[[LOAD]] fastmath<contract> : f64
! CHECK:     %[[CONV:.*]] = fir.convert %[[MUL]] : (f64) -> i32
! CHECK:     acc.yield %[[CONV]] : i32
! CHECK:   }
! CHECK:   acc.atomic.read %[[V_DECL]]#1 = %[[X_DECL]]#1 : !fir.ref<f64>, !fir.ref<i32>, i32
! CHECK: }

subroutine capture_with_convert_i32_to_f32()
  real(4) :: x
  integer :: v
  x = 1.0
  v = 0
  !$acc atomic capture
  v = x
  x = x + v
  !$acc end atomic
end subroutine capture_with_convert_i32_to_f32

! CHECK-LABEL: func.func @_QPcapture_with_convert_i32_to_f32()
! CHECK: %[[V:.*]] = fir.alloca i32 {bindc_name = "v", uniq_name = "_QFcapture_with_convert_i32_to_f32Ev"}
! CHECK: %[[V_DECL:.*]]:2 = hlfir.declare %[[V]] {uniq_name = "_QFcapture_with_convert_i32_to_f32Ev"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[X:.*]] = fir.alloca f32 {bindc_name = "x", uniq_name = "_QFcapture_with_convert_i32_to_f32Ex"}
! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFcapture_with_convert_i32_to_f32Ex"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
! CHECK: hlfir.assign %[[CST]] to %[[X_DECL]]#0 : f32, !fir.ref<f32>
! CHECK: %c0_i32 = arith.constant 0 : i32
! CHECK: hlfir.assign %c0_i32 to %[[V_DECL]]#0 : i32, !fir.ref<i32>
! CHECK: %[[LOAD:.*]] = fir.load %[[V_DECL]]#0 : !fir.ref<i32>
! CHECK: acc.atomic.capture {
! CHECK:   acc.atomic.read %[[V_DECL]]#1 = %[[X_DECL]]#1 : !fir.ref<i32>, !fir.ref<f32>, f32
! CHECK:   acc.atomic.update %[[X_DECL]]#1 : !fir.ref<f32> {
! CHECK:   ^bb0(%arg0: f32):
! CHECK:     %[[CONV:.*]] = fir.convert %[[LOAD]] : (i32) -> f32
! CHECK:     %[[ADD:.*]] = arith.addf %arg0, %[[CONV]] fastmath<contract> : f32
! CHECK:     acc.yield %[[ADD]] : f32
! CHECK:   }
! CHECK: }

subroutine array_ref_in_atomic_capture1
  integer :: x(10), v
  !$acc atomic capture
  v = x(7)
  x(7) = x(7) + 1
  !$acc end atomic
end subroutine array_ref_in_atomic_capture1
! CHECK-LABEL:   func.func @_QParray_ref_in_atomic_capture1() {
! CHECK:           %[[V:.*]] = fir.alloca i32 {bindc_name = "v", uniq_name = "_QFarray_ref_in_atomic_capture1Ev"}
! CHECK:           %[[V_DECL:.*]]:2 = hlfir.declare %[[V]] {uniq_name = "_QFarray_ref_in_atomic_capture1Ev"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[X:.*]] = fir.alloca !fir.array<10xi32> {bindc_name = "x", uniq_name = "_QFarray_ref_in_atomic_capture1Ex"}
! CHECK:           %[[X_DECL:.*]]:2 = hlfir.declare %[[X]](%{{.*}}) {uniq_name = "_QFarray_ref_in_atomic_capture1Ex"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[X_REF:.*]] = hlfir.designate %[[X_DECL]]#0 (%{{.*}})  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
! CHECK:           acc.atomic.capture {
! CHECK:             acc.atomic.read %[[V_DECL]]#1 = %[[X_REF]] : !fir.ref<i32>, !fir.ref<i32>, i32
! CHECK:             acc.atomic.update %[[X_REF]] : !fir.ref<i32> {
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
! CHECK:           %[[V:.*]] = fir.alloca i32 {bindc_name = "v", uniq_name = "_QFarray_ref_in_atomic_capture2Ev"}
! CHECK:           %[[V_DECL:.*]]:2 = hlfir.declare %[[V]] {uniq_name = "_QFarray_ref_in_atomic_capture2Ev"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[X:.*]] = fir.alloca !fir.array<10xi32> {bindc_name = "x", uniq_name = "_QFarray_ref_in_atomic_capture2Ex"}
! CHECK:           %[[X_DECL:.*]]:2 = hlfir.declare %[[X]](%{{.*}}) {uniq_name = "_QFarray_ref_in_atomic_capture2Ex"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK:           %[[X_REF:.*]] = hlfir.designate %[[X_DECL]]#0 (%{{.*}})  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
! CHECK:           acc.atomic.capture {
! CHECK:             acc.atomic.update %[[X_REF]] : !fir.ref<i32> {
! CHECK:             ^bb0(%[[VAL_7:.*]]: i32):
! CHECK:               %[[VAL_8:.*]] = arith.addi %[[VAL_7]], %{{.*}} : i32
! CHECK:               acc.yield %[[VAL_8]] : i32
! CHECK:             }
! CHECK:             acc.atomic.read %[[V_DECL]]#1 = %[[X_REF]] : !fir.ref<i32>, !fir.ref<i32>, i32
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
! CHECK:           %[[V:.*]] = fir.alloca i32 {bindc_name = "v", uniq_name = "_QFcomp_ref_in_atomic_capture1Ev"}
! CHECK:           %[[V_DECL:.*]]:2 = hlfir.declare %[[V]] {uniq_name = "_QFcomp_ref_in_atomic_capture1Ev"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[X:.*]] = fir.alloca !fir.type<_QFcomp_ref_in_atomic_capture1Tt1{c:i32}> {bindc_name = "x", uniq_name = "_QFcomp_ref_in_atomic_capture1Ex"}
! CHECK:           %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFcomp_ref_in_atomic_capture1Ex"} : (!fir.ref<!fir.type<_QFcomp_ref_in_atomic_capture1Tt1{c:i32}>>) -> (!fir.ref<!fir.type<_QFcomp_ref_in_atomic_capture1Tt1{c:i32}>>, !fir.ref<!fir.type<_QFcomp_ref_in_atomic_capture1Tt1{c:i32}>>)
! CHECK:           %[[C:.*]] = hlfir.designate %[[X_DECL]]#0{"c"}   : (!fir.ref<!fir.type<_QFcomp_ref_in_atomic_capture1Tt1{c:i32}>>) -> !fir.ref<i32>
! CHECK:           acc.atomic.capture {
! CHECK:             acc.atomic.read %[[V_DECL]]#1 = %[[C]] : !fir.ref<i32>, !fir.ref<i32>, i32
! CHECK:             acc.atomic.update %[[C]] : !fir.ref<i32> {
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
! CHECK:           %[[V:.*]] = fir.alloca i32 {bindc_name = "v", uniq_name = "_QFcomp_ref_in_atomic_capture2Ev"}
! CHECK:           %[[V_DECL:.*]]:2 = hlfir.declare %[[V]] {uniq_name = "_QFcomp_ref_in_atomic_capture2Ev"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[X:.*]] = fir.alloca !fir.type<_QFcomp_ref_in_atomic_capture2Tt1{c:i32}> {bindc_name = "x", uniq_name = "_QFcomp_ref_in_atomic_capture2Ex"}
! CHECK:           %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFcomp_ref_in_atomic_capture2Ex"} : (!fir.ref<!fir.type<_QFcomp_ref_in_atomic_capture2Tt1{c:i32}>>) -> (!fir.ref<!fir.type<_QFcomp_ref_in_atomic_capture2Tt1{c:i32}>>, !fir.ref<!fir.type<_QFcomp_ref_in_atomic_capture2Tt1{c:i32}>>)
! CHECK:           %[[C:.*]] = hlfir.designate %[[X_DECL]]#0{"c"}   : (!fir.ref<!fir.type<_QFcomp_ref_in_atomic_capture2Tt1{c:i32}>>) -> !fir.ref<i32>
! CHECK:           acc.atomic.capture {
! CHECK:             acc.atomic.update %[[C]] : !fir.ref<i32> {
! CHECK:             ^bb0(%[[VAL_5:.*]]: i32):
! CHECK:               %[[VAL_6:.*]] = arith.addi %[[VAL_5]], %{{.*}} : i32
! CHECK:               acc.yield %[[VAL_6]] : i32
! CHECK:             }
! CHECK:             acc.atomic.read %[[V_DECL]]#1 = %[[C]] : !fir.ref<i32>, !fir.ref<i32>, i32
! CHECK:           }
