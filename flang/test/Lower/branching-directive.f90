!RUN: flang-new -fc1 -emit-hlfir -fopenmp -o - %s | FileCheck %s

!https://github.com/llvm/llvm-project/issues/91526

!CHECK:   cf.cond_br %{{[0-9]+}}, ^bb[[THEN:[0-9]+]], ^bb[[ELSE:[0-9]+]]
!CHECK: ^bb[[THEN]]:
!CHECK:   cf.br ^bb[[EXIT:[0-9]+]]
!CHECK: ^bb[[ELSE]]:
!CHECK:   fir.call @_FortranAStopStatement
!CHECK:   fir.unreachable
!CHECK: ^bb[[EXIT]]:

subroutine simple(y)
  implicit none
  logical, intent(in) :: y
  integer :: i
  if (y) then
!$omp parallel
    i = 1
!$omp end parallel
  else
    stop 1
  end if
end subroutine simple

