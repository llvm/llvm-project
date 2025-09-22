! This test checks lowering of grainsize clause in taskloop directive.

! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s

! CHECK-LABEL:  omp.private
! CHECK-SAME:       {type = private} @[[I_PRIVATE_TEST2:.*]] : i32

! CHECK-LABEL:  omp.private
! CHECK-SAME:        {type = firstprivate} @[[X_FIRSTPRIVATE_TEST2:.*]] : i32
! CHECK-SAME:   copy {
! CHECK:         hlfir.assign

! CHECK-LABEL:  omp.private
! CHECK-SAME:       {type = private} @[[I_PRIVATE:.*]] : i32

! CHECK-LABEL:  omp.private
! CHECK-SAME:        {type = firstprivate} @[[X_FIRSTPRIVATE:.*]] : i32
! CHECK-SAME:   copy {
! CHECK:         hlfir.assign

! CHECK-LABEL:  func.func @_QPtest_grainsize
! CHECK:          %[[ALLOCA_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_grainsizeEi"}
! CHECK:          %[[DECL_I:.*]]:2 = hlfir.declare %[[ALLOCA_I]] {uniq_name = "_QFtest_grainsizeEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:          %[[ALLOCA_X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFtest_grainsizeEx"}
! CHECK:          %[[DECL_X:.*]]:2 = hlfir.declare %[[ALLOCA_X]] {uniq_name = "_QFtest_grainsizeEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:          %[[GRAINSIZE:.*]] = arith.constant 10 : i32
subroutine test_grainsize
   integer :: i, x
   ! CHECK:          omp.taskloop grainsize(%[[GRAINSIZE]]: i32) 
   ! CHECK-SAME:        private(@[[X_FIRSTPRIVATE]] %[[DECL_X]]#0 -> %[[ARG0:.*]], @[[I_PRIVATE]] %[[DECL_I]]#0 -> %[[ARG1:.*]] : !fir.ref<i32>, !fir.ref<i32>) {
   ! CHECK:            omp.loop_nest (%[[ARG2:.*]]) : i32 = (%{{.*}}) to (%{{.*}}) inclusive step (%{{.*}}) {
   !$omp taskloop grainsize(10)
   do i = 1, 1000
      x = x + 1
   end do
   !$omp end taskloop
end subroutine test_grainsize

!CHECK-LABEL: func.func @_QPtest_grainsize_strict()
subroutine test_grainsize_strict
  integer :: i, x
  ! CHECK: %[[GRAINSIZE:.*]] = arith.constant 10 : i32
  ! CHECK: omp.taskloop grainsize(strict, %[[GRAINSIZE]]: i32)
  !$omp taskloop grainsize(strict:10)
  do i = 1, 1000
     !CHECK: arith.addi
     x = x + 1
  end do
  !$omp end taskloop
end subroutine
