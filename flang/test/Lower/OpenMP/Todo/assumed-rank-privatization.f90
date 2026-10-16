! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: Privatization of assumed rank variable
subroutine assumedPriv(a)
  integer :: a(..)

  !$omp parallel private(a)
  !$omp end parallel
end
