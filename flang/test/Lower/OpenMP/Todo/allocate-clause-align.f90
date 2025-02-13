! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -fopenmp-version=51 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: OmpAllocateClause ALIGN modifier
program p
  integer :: x
  integer :: a
  integer :: i
  !$omp parallel private(x) allocate(align(4): x)
  do i=1,10
     a = a + i
  end do
  !$omp end parallel
end program p
