! An otherwise-undeclared acc routine bind(name) target still gets a func.func.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! CHECK: acc.routine @[[ACLEAR_SEQ_ROUTINE:.*]] func(@_QPaclear_seq) seq
! CHECK: acc.routine @{{.*}} func(@_QPaclear) bind(@_QPaclear_seq) seq
! CHECK: func.func private @_QPaclear_seq({{.*}}) attributes {acc.routine_info = #acc.routine_info<[@[[ACLEAR_SEQ_ROUTINE]]]>}

subroutine s_bind_undeclared(n, x)
  integer :: n, i
  real :: x(n)
  !$acc routine(aclear) seq bind(aclear_seq)
  external :: aclear
  !$acc parallel loop
  do i = 1, n
    call aclear(x(i))
  end do
end subroutine
