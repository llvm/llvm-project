! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: OpenMP reduction allocation for dynamic length character
subroutine test_dynamic_length(n)
  integer, intent(in) :: n
  character(len=n) :: var

  !$omp declare reduction (char_max_dyn:character(len=n):omp_out=max(omp_out,omp_in)) &
  !$omp initializer(omp_priv='a')
end subroutine test_dynamic_length
