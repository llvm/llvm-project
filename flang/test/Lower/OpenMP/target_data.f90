!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s

!===============================================================================
! Target_Enter Simple
!===============================================================================


!CHECK-LABEL: func.func @_QPomp_target_enter_simple() {
subroutine omp_target_enter_simple
   integer :: a(1024)
   !CHECK: omp.target_enter_data   map((to -> {{.*}} : !fir.ref<!fir.array<1024xi32>>))
   !$omp target enter data map(to: a)
end subroutine omp_target_enter_simple

!===============================================================================
! Target_Enter Map types
!===============================================================================

!CHECK-LABEL: func.func @_QPomp_target_enter_mt() {
subroutine omp_target_enter_mt
   integer :: a(1024)
   integer :: b(1024)
   integer :: c(1024)
   integer :: d(1024)
   !CHECK: omp.target_enter_data   map((to -> {{.*}} : !fir.ref<!fir.array<1024xi32>>), (to -> {{.*}} : !fir.ref<!fir.array<1024xi32>>), (always, alloc -> {{.*}} : !fir.ref<!fir.array<1024xi32>>), (to -> {{.*}} : !fir.ref<!fir.array<1024xi32>>))
   !$omp target enter data map(to: a, b) map(always, alloc: c) map(to: d)
end subroutine omp_target_enter_mt

!===============================================================================
! `Nowait` clause
!===============================================================================

!CHECK-LABEL: func.func @_QPomp_target_enter_nowait() {
subroutine omp_target_enter_nowait
   integer :: a(1024)
   !CHECK: omp.target_enter_data   nowait map((to -> {{.*}} : !fir.ref<!fir.array<1024xi32>>))
   !$omp target enter data map(to: a) nowait
end subroutine omp_target_enter_nowait

!===============================================================================
! `if` clause
!===============================================================================

!CHECK-LABEL: func.func @_QPomp_target_enter_if() {
subroutine omp_target_enter_if
   integer :: a(1024)
   integer :: i
   i = 5
   !CHECK: %[[VAL_3:.*]] = fir.load %[[VAL_1:.*]] : !fir.ref<i32>
   !CHECK: %[[VAL_4:.*]] = arith.constant 10 : i32
   !CHECK: %[[VAL_5:.*]] = arith.cmpi slt, %[[VAL_3:.*]], %[[VAL_4:.*]] : i32
   !CHECK: omp.target_enter_data   if(%[[VAL_5:.*]] : i1) map((to -> {{.*}} : !fir.ref<!fir.array<1024xi32>>))
   !$omp target enter data if(i<10) map(to: a)
end subroutine omp_target_enter_if

!===============================================================================
! `device` clause
!===============================================================================

!CHECK-LABEL: func.func @_QPomp_target_enter_device() {
subroutine omp_target_enter_device
   integer :: a(1024)
   !CHECK: %[[VAL_1:.*]] = arith.constant 2 : i32
   !CHECK: omp.target_enter_data   device(%[[VAL_1:.*]] : i32) map((to -> {{.*}} : !fir.ref<!fir.array<1024xi32>>))
   !$omp target enter data map(to: a) device(2)
end subroutine omp_target_enter_device

!===============================================================================
! Target_Exit Simple
!===============================================================================

!CHECK-LABEL: func.func @_QPomp_target_exit_simple() {
subroutine omp_target_exit_simple
   integer :: a(1024)
   !CHECK: omp.target_exit_data   map((from -> {{.*}} : !fir.ref<!fir.array<1024xi32>>))
   !$omp target exit data map(from: a)
end subroutine omp_target_exit_simple

!===============================================================================
! Target_Exit Map types
!===============================================================================

!CHECK-LABEL: func.func @_QPomp_target_exit_mt() {
subroutine omp_target_exit_mt
   integer :: a(1024)
   integer :: b(1024)
   integer :: c(1024)
   integer :: d(1024)
   integer :: e(1024)
   !CHECK: omp.target_exit_data   map((from -> {{.*}} : !fir.ref<!fir.array<1024xi32>>), (from -> {{.*}} : !fir.ref<!fir.array<1024xi32>>), (release -> {{.*}} : !fir.ref<!fir.array<1024xi32>>), (always, delete -> {{.*}} : !fir.ref<!fir.array<1024xi32>>), (from -> {{.*}} : !fir.ref<!fir.array<1024xi32>>))
   !$omp target exit data map(from: a,b) map(release: c) map(always, delete: d) map(from: e)
end subroutine omp_target_exit_mt

!===============================================================================
! `device` clause
!===============================================================================

!CHECK-LABEL: func.func @_QPomp_target_exit_device() {
subroutine omp_target_exit_device
   integer :: a(1024)
   integer :: d
   !CHECK: %[[VAL_2:.*]] = fir.load %[[VAL_1:.*]] : !fir.ref<i32>
   !CHECK: omp.target_exit_data   device(%[[VAL_2:.*]] : i32) map((from -> {{.*}} : !fir.ref<!fir.array<1024xi32>>))
   !$omp target exit data map(from: a) device(d)
end subroutine omp_target_exit_device
