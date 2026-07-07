! Test that a metadirective variant resolving to a target construct
! correctly reports a TODO (host-eval support needed).

! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: TARGET construct selected by METADIRECTIVE (host-eval)

subroutine test_target_loop()
  integer :: i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: target teams distribute parallel do) &
  !$omp & default(nothing)
  do i = 1, 100
  end do
end subroutine
