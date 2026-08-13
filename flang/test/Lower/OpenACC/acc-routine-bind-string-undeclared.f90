! A bind("name") target is its own external name, declared verbatim (not
! mangled), so later passes reference a live symbol.

! RUN: bbc -fopenacc -emit-hlfir --mlir-print-debuginfo --mlir-print-local-scope %s -o - | FileCheck %s

! CHECK: acc.routine @[[ACLEAR_DEV_ROUTINE:.*]] func(@aclear_dev) seq
! CHECK: acc.routine @{{.*}} func(@_QPaclear) bind("aclear_dev") seq
! CHECK: func.func private @aclear_dev({{.*}}) attributes {acc.routine_info = #acc.routine_info<[@[[ACLEAR_DEV_ROUTINE]]]>}
! CHECK-SAME: loc("{{.*}}acc-routine-bind-string-undeclared.f90":{{[0-9]+}}:{{[0-9]+}})
! CHECK-NOT: func.func private @aclear_dev{{.*}}loc(unknown)

subroutine s_bind_str_undeclared(n, x)
  integer :: n, i
  real :: x(n)
  !$acc routine(aclear) seq bind("aclear_dev")
  external :: aclear
  !$acc parallel loop
  do i = 1, n
    call aclear(x(i))
  end do
end subroutine
