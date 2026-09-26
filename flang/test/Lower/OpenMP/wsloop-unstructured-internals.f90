! RUN: bbc -fopenmp -emit-hlfir -o - %s | FileCheck %s

! A loop associated with an OpenMP loop directive whose branching is confined
! to its body. The directive's code-gen consumes the DO, so the body is wrapped
! at the directive's body lowering site rather than in genFIR(DoConstruct). The
! loop stays a single omp.loop_nest -- it is not dissolved into a cf trip-count
! loop -- so it is still available to be worksharded.

! A forward GOTO raised inside a nested IF, jumping over a whole inner DO and
! landing on the last statement of the outer loop body. Both endpoints are in
! the body, so the associated loop keeps its structured form and the two inner
! loops stay plain fir.do_loops inside the wrap.
subroutine omp_goto_over_inner(qfx, a, n, force, flux)
  real :: qfx(n,n), a(n,n)
  logical :: force
  integer :: flux
  !$omp parallel do
  do j = 1, n
    do 330 i = 1, n
      a(i,j) = a(i,j) + 1.0
330 continue
    if (force) then
      if (flux .eq. 1) goto 350
    endif
    do i = 1, n
      qfx(i,j) = 0.
    enddo
350 continue
  enddo
  !$omp end parallel do
end subroutine

! CHECK-LABEL: func.func @_QPomp_goto_over_inner
! CHECK:         omp.wsloop
! CHECK:           omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) inclusive step (%{{.*}}) {
! CHECK:             scf.execute_region no_inline {
! The inner loops are untouched by the wrap around the outer body.
! CHECK:               fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} : i32 {
! The GOTO and the branches it feeds stay inside the region.
! CHECK:               cf.cond_br %{{[0-9]+}}, ^bb[[THEN:[0-9]+]], ^bb[[SKIP:[0-9]+]]
! CHECK:               cf.cond_br %{{[0-9]+}}, ^bb[[GOTO:[0-9]+]], ^bb[[SKIP]]
! CHECK:               fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} : i32 {
! CHECK:               scf.yield
! CHECK:             omp.yield
