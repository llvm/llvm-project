! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -fopenmp-version=51 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: OmpDeclareVariantDirective

subroutine sb1
  integer :: x
  x = 1
  call sub(x)
contains
  subroutine vsub (v1)
    integer, value :: v1
  end
  subroutine sub (v1)
    !$omp declare variant(vsub), match(construct={dispatch})
    integer, value :: v1
  end
end subroutine
