! REQUIRES: openmp_runtime
! RUN: %not_todo_cmd %flang_fc1 -emit-llvm %openmp_flags -fopenmp-version=51 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: Unhandled clause allocate in omp.parallel
! CHECK: LLVM Translation failed for operation: omp.parallel
program p
  use omp_lib
  integer :: x
  integer :: a
  integer :: i
  !$omp parallel private(x) allocate(allocator(omp_default_mem_alloc): x)
  do i=1,10
     a = a + i
  end do
  !$omp end parallel
end program p
