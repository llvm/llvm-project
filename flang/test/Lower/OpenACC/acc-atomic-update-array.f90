! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

subroutine atomic_update_array1(r, n, x)
  implicit none
  integer :: n
  real :: r(n), x
  integer :: i
   
  !$acc data copy(r)

  !$acc parallel loop
  do i = 1, n
    !$acc atomic update
    r(i) = r(i) + x
    !$acc end atomic
  end do

  !$acc end data
end subroutine

! CHECK-LABEL: func.func @_QPatomic_update_array1(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "r"}, %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}, %[[ARG2:.*]]: !fir.ref<f32> {fir.bindc_name = "x"}) {
! CHECK: %[[DECL_ARG2:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFatomic_update_array1Ex"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK: %[[DECL_ARG0:.*]]:2 = hlfir.declare %[[ARG0]](%{{.*}}) dummy_scope %{{[0-9]+}} {uniq_name = "_QFatomic_update_array1Er"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! CHECK: %[[ARRAY_REF:.*]] = hlfir.designate %[[DECL_ARG0]]#0 (%{{.*}})  : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
! CHECK: %[[LOAD_X:.*]] = fir.load %[[DECL_ARG2]]#0 : !fir.ref<f32>
! CHECK: acc.atomic.update %[[ARRAY_REF]] : !fir.ref<f32> {
! CHECK: ^bb0(%[[ARG:.*]]: f32):
! CHECK:   %[[ATOMIC:.*]] = arith.addf %[[ARG]], %[[LOAD_X]] fastmath<contract> : f32
! CHECK:   acc.yield %[[ATOMIC]] : f32
! CHECK: }


subroutine atomic_read_array1(r, n, x)
  implicit none
  integer :: n
  real :: r(n), x

  !$acc atomic read
  x = r(n)
end subroutine

! CHECK-LABEL: func.func @_QPatomic_read_array1(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "r"}, %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}, %[[ARG2:.*]]: !fir.ref<f32> {fir.bindc_name = "x"}) {
! CHECK: %[[DECL_X:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFatomic_read_array1Ex"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK: %[[DECL_R:.*]]:2 = hlfir.declare %[[ARG0]](%{{.*}}) dummy_scope %{{[0-9]+}} {uniq_name = "_QFatomic_read_array1Er"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! CHECK: %[[DES:.*]] = hlfir.designate %[[DECL_R]]#0 (%{{.*}})  : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
! CHECK: acc.atomic.read %[[DECL_X]]#1 = %[[DES]] : !fir.ref<f32>, !fir.ref<f32>, f32

subroutine atomic_write_array1(r, n, x)
  implicit none
  integer :: n
  real :: r(n), x
  
  !$acc atomic write
  x = r(n)
end subroutine

! CHECK-LABEL: func.func @_QPatomic_write_array1(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "r"}, %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}, %[[ARG2:.*]]: !fir.ref<f32> {fir.bindc_name = "x"}) {
! CHECK: %[[DECL_X:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFatomic_write_array1Ex"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK: %[[DECL_R:.*]]:2 = hlfir.declare %[[ARG0]](%{{.*}}) dummy_scope %{{[0-9]+}} {uniq_name = "_QFatomic_write_array1Er"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! CHECK: %[[DES:.*]] = hlfir.designate %[[DECL_R]]#0 (%{{.*}})  : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
! CHECK: %[[LOAD:.*]] = fir.load %[[DES]] : !fir.ref<f32> 
! CHECK: acc.atomic.write %[[DECL_X]]#1 = %[[LOAD]] : !fir.ref<f32>, f32

subroutine atomic_capture_array1(r, n, x, y)
  implicit none
  integer :: n, i
  real :: r(n), x, y

  !$acc atomic capture
  r(i) = r(i) + x
  y = r(i)
  !$acc end atomic
end subroutine

! CHECK-LABEL: func.func @_QPatomic_capture_array1(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "r"}, %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}, %[[ARG2:.*]]: !fir.ref<f32> {fir.bindc_name = "x"}, %[[ARG3:.*]]: !fir.ref<f32> {fir.bindc_name = "y"}) {
! CHECK: %[[DECL_X:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFatomic_capture_array1Ex"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK: %[[DECL_Y:.*]]:2 = hlfir.declare %[[ARG3]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFatomic_capture_array1Ey"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK: %[[DECL_R:.*]]:2 = hlfir.declare %[[ARG0]](%{{.*}}) dummy_scope %{{[0-9]+}} {uniq_name = "_QFatomic_capture_array1Er"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! CHECK: %[[R_I:.*]] = hlfir.designate %[[DECL_R]]#0 (%{{.*}})  : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
! CHECK: %[[LOAD:.*]] = fir.load %[[DECL_X]]#0 : !fir.ref<f32>
! CHECK: acc.atomic.capture {
! CHECK:   acc.atomic.update %[[R_I]] : !fir.ref<f32> {
! CHECK:   ^bb0(%[[ARG:.*]]: f32):
! CHECK:     %[[ADD:.*]] = arith.addf %[[ARG]], %[[LOAD]] fastmath<contract> : f32
! CHECK:     acc.yield %[[ADD]] : f32
! CHECK:   }
! CHECK:   acc.atomic.read %[[DECL_Y]]#1 = %[[R_I]] : !fir.ref<f32>, !fir.ref<f32>, f32
! CHECK: }
