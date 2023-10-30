! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s --check-prefixes=CHECK,FIR
! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK,HLFIR

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
! HLFIR: %[[DECL_ARG2:.*]]:2 = hlfir.declare %[[ARG2]] {uniq_name = "_QFatomic_update_array1Ex"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! HLFIR: %[[DECL_ARG0:.*]]:2 = hlfir.declare %[[ARG0]](%{{.*}}) {uniq_name = "_QFatomic_update_array1Er"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! FIR:   %[[ARRAY_REF:.*]] = fir.coordinate_of %[[ARG0]], %{{.*}} : (!fir.ref<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
! HLFIR: %[[ARRAY_REF:.*]] = hlfir.designate %[[DECL_ARG0]]#0 (%{{.*}})  : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
! FIR:   %[[LOAD_X:.*]] = fir.load %[[ARG2]] : !fir.ref<f32>
! HLFIR: %[[LOAD_X:.*]] = fir.load %[[DECL_ARG2]]#0 : !fir.ref<f32>
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
! FIR: %[[ARRAY_REF:.*]] = fir.coordinate_of %[[ARG0]], %{{.*}} : (!fir.ref<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
! FIR: acc.atomic.read %[[ARG2]] = %[[ARRAY_REF]] : !fir.ref<f32>, f32
! HLFIR: %[[DECL_X:.*]]:2 = hlfir.declare %[[ARG2]] {uniq_name = "_QFatomic_read_array1Ex"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! HLFIR: %[[DECL_R:.*]]:2 = hlfir.declare %[[ARG0]](%{{.*}}) {uniq_name = "_QFatomic_read_array1Er"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! HLFIR: %[[DES:.*]] = hlfir.designate %[[DECL_R]]#0 (%{{.*}})  : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
! HLFIR: acc.atomic.read %[[DECL_X]]#1 = %[[DES]] : !fir.ref<f32>, f32

subroutine atomic_write_array1(r, n, x)
  implicit none
  integer :: n
  real :: r(n), x
  
  !$acc atomic write
  x = r(n)
end subroutine

! CHECK-LABEL: func.func @_QPatomic_write_array1(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "r"}, %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}, %[[ARG2:.*]]: !fir.ref<f32> {fir.bindc_name = "x"}) {
! FIR: %[[ARRAY_REF:.*]] = fir.coordinate_of %[[ARG0]], %{{.*}} : (!fir.ref<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
! FIR: %[[LOAD:.*]] = fir.load %[[ARRAY_REF]] : !fir.ref<f32> 
! FIR: acc.atomic.write %[[ARG2]] = %[[LOAD]] : !fir.ref<f32>, f32
! HLFIR: %[[DECL_X:.*]]:2 = hlfir.declare %[[ARG2]] {uniq_name = "_QFatomic_write_array1Ex"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! HLFIR: %[[DECL_R:.*]]:2 = hlfir.declare %[[ARG0]](%{{.*}}) {uniq_name = "_QFatomic_write_array1Er"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! HLFIR: %[[DES:.*]] = hlfir.designate %[[DECL_R]]#0 (%{{.*}})  : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
! HLFIR: %[[LOAD:.*]] = fir.load %[[DES]] : !fir.ref<f32> 
! HLFIR: acc.atomic.write %[[DECL_X]]#1 = %[[LOAD]] : !fir.ref<f32>, f32

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
! HLFIR: %[[DECL_X:.*]]:2 = hlfir.declare %[[ARG2]] {uniq_name = "_QFatomic_capture_array1Ex"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! HLFIR: %[[DECL_Y:.*]]:2 = hlfir.declare %[[ARG3]] {uniq_name = "_QFatomic_capture_array1Ey"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! HLFIR: %[[DECL_R:.*]]:2 = hlfir.declare %[[ARG0]](%{{.*}}) {uniq_name = "_QFatomic_capture_array1Er"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! HLFIR: %[[R_I:.*]] = hlfir.designate %[[DECL_R]]#0 (%{{.*}})  : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
! FIR:   %[[R_I:.*]] = fir.coordinate_of %[[ARG0]], %{{.*}} : (!fir.ref<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
! HLFIR: %[[LOAD:.*]] = fir.load %[[DECL_X]]#0 : !fir.ref<f32>
! FIR:   %[[LOAD:.*]] = fir.load %[[ARG2]] : !fir.ref<f32>
! CHECK: acc.atomic.capture {
! CHECK:   acc.atomic.update %[[R_I]] : !fir.ref<f32> {
! CHECK:   ^bb0(%[[ARG:.*]]: f32):
! CHECK:     %[[ADD:.*]] = arith.addf %[[ARG]], %[[LOAD]] fastmath<contract> : f32
! CHECK:     acc.yield %[[ADD]] : f32
! CHECK:   }
! HLFIR:   acc.atomic.read %[[DECL_Y]]#1 = %[[R_I]] : !fir.ref<f32>, f32
! FIR:     acc.atomic.read %[[ARG3]] = %[[R_I]] : !fir.ref<f32>, f32
! CHECK: }
