! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -o - %s | FileCheck %s

subroutine test_unroll_full
  integer res, i

  !$omp unroll full
  do i = 1, 100
    res = i
  end do
  !$omp end unroll
end subroutine test_unroll_full

! CHECK-LABEL: func.func @_QPtest_unroll_full() {
! CHECK:         %[[CLI:.+]] = omp.new_cli
! CHECK:         omp.canonical_loop(%[[CLI]]) %{{.*}} : i32 in range(%{{.*}}) {
! CHECK:         omp.unroll_full(%[[CLI]])
! CHECK:       }
