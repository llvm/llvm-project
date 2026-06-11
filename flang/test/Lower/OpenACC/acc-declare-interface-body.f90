! Check that an acc declare in an interface body is not hoisted into the
! enclosing program unit and lowered there (which crashed lowering), while the
! actual procedure definition still lowers its own declare.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

program test_acc_declare_interface
  implicit none
  integer, parameter :: n1 = 10, nlev = 60
  real, dimension(n1, nlev) :: a

  interface
    subroutine compute(n1, nlev, a)
      integer :: n1, nlev
      real, dimension(n1, nlev) :: a
!$acc declare present(a)
    end subroutine compute
  end interface

  a = 0.1
  call compute(n1, nlev, a)
end program

subroutine compute(n1, nlev, a)
  integer :: n1, nlev
  real, dimension(n1, nlev) :: a
!$acc declare present(a)
  a = a * a
end subroutine compute

! The interface-body declare is not hoisted into the host, so nothing is lowered.
! CHECK-LABEL: func.func @_QQmain()
! CHECK-NOT:     acc.declare_enter
! CHECK-NOT:     acc.present
! CHECK:         fir.call @_QPcompute
! CHECK:         return

! The actual definition lowers the present declare on its dummy argument.
! CHECK-LABEL: func.func @_QPcompute(
! CHECK:         %[[DECL:.*]]:2 = hlfir.declare %{{.*}} {acc.declare = #acc.declare<dataClause = acc_present>, uniq_name = "_QFcomputeEa"}
! CHECK:         %[[PRES:.*]] = acc.present var(%[[DECL]]#0
! CHECK:         acc.declare_enter dataOperands(%[[PRES]]
! CHECK:         acc.declare_exit
