! A bind(name) target with no declaration of its own is declared by cloning the
! decorated routine's function type (not () -> ()), since bind renames the same
! callable. The cloned declaration carries the type only (no argument attributes).

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

subroutine s_bind_clone(n, x)
  integer :: n, i
  real :: x(n)
  interface
    subroutine aclear(y)
      real :: y(:)
    end subroutine
  end interface
  !$acc routine(aclear) seq bind(aclear_seq)
  !$acc parallel loop
  do i = 1, n
    call aclear(x(i:i))
  end do
end subroutine

! CHECK: acc.routine @[[ACLEAR_SEQ_ROUTINE:.*]] func(@_QPaclear_seq) seq
! CHECK: acc.routine @{{.*}} func(@_QPaclear) bind(@_QPaclear_seq) seq
! The decorated routine's signature (assumed-shape array descriptor):
! CHECK: func.func private @_QPaclear(!fir.box<!fir.array<?xf32>>
! The bind target is declared with the same type, proving the clone:
! CHECK: func.func private @_QPaclear_seq(!fir.box<!fir.array<?xf32>>) attributes {acc.routine_info = #acc.routine_info<[@[[ACLEAR_SEQ_ROUTINE]]]>}
