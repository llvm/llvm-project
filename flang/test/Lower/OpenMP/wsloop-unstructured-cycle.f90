! RUN: bbc --wrap-unstructured-constructs-in-execute-region -emit-hlfir -fopenmp -o - %s | FileCheck %s --implicit-check-not=scf.execute_region

! A DO associated with an OpenMP loop directive is lowered by the directive's
! own code-gen. Such a DO must never be folded into an
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

! COLLAPSE(n) and ORDERED(n) both associate n loops with the directive, and
! the loop transforming directives (TILE, INTERCHANGE, ...) associate as many
! as their arguments describe. None of the associated loops may be wrapped.

subroutine collapse_case(x, y, n)
  implicit none
  integer n
  double precision x(*), y(*)
  integer i, j

  !$omp do collapse(2)
  do i = 1, n
    do j = 1, n
      if (x(i) > 0.0d0) then
        y(1) = 0.0d0
        cycle
      end if
      y(2) = 1.0d0
    end do
  end do
  !$omp end do

end subroutine collapse_case

! Both loops are associated with the directive, so the body stays flat inside
! omp.loop_nest.
! CHECK-LABEL: func.func @_QPcollapse_case(
! CHECK:         omp.wsloop
! CHECK:           omp.loop_nest ({{.*}}) {{.*}} collapse(2) {
! CHECK:             cf.cond_br
! CHECK:             omp.yield

subroutine ordered_case(x, y, n)
  implicit none
  integer n
  double precision x(*), y(*)
  integer i, j

  !$omp do ordered(2)
  do i = 1, n
    do j = 1, n
      if (x(i) > 0.0d0) then
        y(1) = 0.0d0
        cycle
      end if
      y(2) = 1.0d0
    end do
  end do
  !$omp end do

end subroutine ordered_case

! ORDERED(2) associates the inner loop with the directive as well, so it is
! not wrapped either.
! CHECK-LABEL: func.func @_QPordered_case(
! CHECK:         omp.wsloop ordered(2)
! CHECK:           omp.loop_nest
! CHECK:             cf.cond_br
! CHECK:             omp.yield

! A TILE case belongs here too, since the SIZES arguments decide how many
! loops are associated with the directive. It is left out for now because
! lowering a TILE whose body is unstructured currently fails an assertion,
! independently of whether wrapping is enabled:
! https://github.com/llvm/llvm-project/issues/216701
