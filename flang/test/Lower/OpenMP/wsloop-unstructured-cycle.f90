! RUN: bbc --wrap-unstructured-constructs-in-execute-region -emit-hlfir -fopenmp -o - %s | FileCheck %s

! A DO associated with an OpenMP loop directive is lowered by the directive's
! own code-gen, which never reaches genFIR(DoConstruct) where a plain loop's
! body is wrapped. The body is wrapped at the directive's own body-lowering
! site instead, so a loop whose branching is confined to its body -- here an
! IF-guarded CYCLE -- keeps its structured form inside omp.loop_nest.

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

! The CYCLE targets the EndDoStmt, which inside the wrap is the region's yield
! block -- so it leaves the region instead of branching to a block outside it.
! CHECK-LABEL: func.func @_QPrepro_final(
! CHECK:         omp.wsloop
! CHECK:           omp.loop_nest
! CHECK:             hlfir.assign
! CHECK:             scf.execute_region no_inline {
! CHECK:             ^bb[[TEST:[0-9]+]]:
! CHECK:               arith.cmpf ogt
! CHECK:               cf.cond_br %{{[0-9]+}}, ^bb[[CYCLE:[0-9]+]], ^bb[[BODY:[0-9]+]]
! CHECK:             ^bb[[CYCLE]]:
! CHECK:               hlfir.assign
! CHECK:               cf.br ^bb[[EXIT:[0-9]+]]
! CHECK:             ^bb[[BODY]]:
! CHECK:               hlfir.assign
! CHECK:               cf.br ^bb[[EXIT]]
! CHECK:             ^bb[[EXIT]]:
! CHECK:               scf.yield
! CHECK:             omp.yield

! COLLAPSE(n) and ORDERED(n) both associate n loops with the directive. The
! associated loops are collapsed into a single omp.loop_nest, and the wrap goes
! around the innermost body it encloses.

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

! Both loops are associated with the directive, so one wrap covers the body of
! the collapsed nest.
! CHECK-LABEL: func.func @_QPcollapse_case(
! CHECK:         omp.wsloop
! CHECK:           omp.loop_nest ({{.*}}) {{.*}} collapse(2) {
! CHECK:             scf.execute_region no_inline {
! CHECK:               cf.cond_br
! CHECK:               scf.yield
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

! ORDERED(2) keeps the inner loop as a loop of its own inside the nest, so the
! outer body and the inner body each get a wrap.
! CHECK-LABEL: func.func @_QPordered_case(
! CHECK:         omp.wsloop ordered(2)
! CHECK:           omp.loop_nest
! CHECK:             scf.execute_region no_inline {
! CHECK:               scf.execute_region no_inline {
! CHECK:                 cf.cond_br
! CHECK:                 scf.yield
! CHECK:               scf.yield
! CHECK:             omp.yield

! A TILE case belongs here too, since the SIZES arguments decide how many
! loops are associated with the directive. It is left out for now because
! lowering a TILE whose body is unstructured currently fails an assertion,
! independently of whether wrapping is enabled:
! https://github.com/llvm/llvm-project/issues/216701
