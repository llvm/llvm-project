!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! This test checks that if reduction clause is on a combined target
! construct, there is an implicit map(tofrom) for each reduction variable.

! construct with target
subroutine omp_target_combined
   implicit none
   integer(kind = 8) :: s1
   integer(kind = 8) :: s2
   integer(kind = 4) ::  i
   s1 = 1
   s2 = 1
   !$omp target teams distribute parallel do reduction(+:s1) reduction(+:s2)
      do i=1,1000
          s1 = s1 + i
          s2 = s2 + i
      end do
   !$omp end target teams distribute parallel do
   return
end subroutine omp_target_combined
!CHECK-LABEL: func.func @_QPomp_target_combined() {
!CHECK: omp.map.info var_ptr({{.*}} : !fir.ref<i64>, i64) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i64> {name = "s1"}
!CHECK: omp.map.info var_ptr({{.*}} : !fir.ref<i64>, i64) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i64> {name = "s2"}
!CHECK: omp.map.info var_ptr({{.*}} : !fir.ref<i32>, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !fir.ref<i32> {name = "i"}

subroutine omp_target_team_separate
   implicit none
   integer(kind = 8) :: s3
   integer i
   s3 = 1
   !$omp target
   s3 = 2
   !$omp teams distribute parallel do reduction(+:s3)
      do i=1,1000
         s3 = s3 + i
      end do
   !$omp end teams distribute parallel do
   !$omp end target
   return
end subroutine omp_target_team_separate
!CHECK-LABEL: func.func @_QPomp_target_team_separate() {
!CHECK:  omp.map.info var_ptr({{.*}} : !fir.ref<i64>, i64) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !fir.ref<i64> {name = "s3"}
