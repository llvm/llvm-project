! RUN: bbc --wrap-unstructured-constructs-in-execute-region -emit-hlfir -fopenmp -o - %s | FileCheck %s --implicit-check-not=scf.execute_region

! A DO attached to an OpenMP loop directive is lowered by the directive's own
! code-gen, which takes the loop over. Such a DO must never be folded into an
! scf.execute_region, even when wrapping is enabled and the loop is
! unstructured -- here the IF-guarded CYCLE makes it so. The body's blocks
! stay flat inside omp.loop_nest.
!
! --implicit-check-not on the RUN line asserts that no wrapping takes place
! anywhere in the output.

subroutine repro_final(x, y, n)
  implicit none
  integer n
  double precision x(*), y(*)
  integer i

  !$omp do
  do i = 1, n
    if (x(i) > 0.0d0) then
      y(1) = 0.0d0   ! any statement before CYCLE makes the loop unstructured
      cycle
    end if
    y(2) = 1.0d0
  end do
  !$omp end do

end subroutine repro_final

! CHECK-LABEL: func.func @_QPrepro_final(
! CHECK:         omp.wsloop
! CHECK:           omp.loop_nest
! CHECK:             hlfir.assign
! CHECK:             cf.br ^bb[[TEST:[0-9]+]]
! CHECK:           ^bb[[TEST]]:
! CHECK:             arith.cmpf ogt
! CHECK:             cf.cond_br %{{[0-9]+}}, ^bb[[CYCLE:[0-9]+]], ^bb[[BODY:[0-9]+]]
! CHECK:           ^bb[[CYCLE]]:
! CHECK:             hlfir.assign
! CHECK:             cf.br ^bb[[EXIT:[0-9]+]]
! CHECK:           ^bb[[BODY]]:
! CHECK:             hlfir.assign
! CHECK:             cf.br ^bb[[EXIT]]
! CHECK:           ^bb[[EXIT]]:
! CHECK:             omp.yield
