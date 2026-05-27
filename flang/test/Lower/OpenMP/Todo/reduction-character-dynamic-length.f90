! RUN: %not_todo_cmd bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck -check-prefix=BBC %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck -check-prefix=FC1 %s

! BBC: not yet implemented: declare reduction currently only supports trivial types, fixed-length CHARACTER, or derived types containing them
! FC1: not yet implemented: declare reduction currently only supports trivial types, fixed-length CHARACTER, or derived types containing them
subroutine test_dynamic_length(n)
  integer, intent(in) :: n
  character(len=n) :: var

  !$omp declare reduction (char_max_dyn:character(len=n):omp_out=max(omp_out,omp_in)) &
  !$omp initializer(omp_priv='a')
end subroutine test_dynamic_length
