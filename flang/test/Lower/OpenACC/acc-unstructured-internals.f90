! RUN: bbc -fopenacc -emit-hlfir -o - %s | FileCheck %s

! Loops under an OpenACC directive whose control flow is structured on the
! outside but whose branching is confined to the body. The directive's own
! code-gen consumes the DO, so the body is wrapped at the directive's body
! lowering site rather than in genFIR(DoConstruct). The loop keeps its bounds
! on the op -- control(...) rather than a cf trip-count test -- so it is still
! available to be parallelized.

! A forward GOTO raised inside a nested IF, jumping over a whole inner DO and
! landing on the last statement of the outer loop body. Both endpoints are
! inside the body, so the outer loop stays an acc.loop with control(...) and
! the two inner loops stay acc.loops of their own.
subroutine kernels_goto_over_inner(qfx, a, its, ite, jts, jte, force, flux)
  real :: qfx(ite,jte), a(ite,jte)
  logical :: force
  integer :: flux
  !$acc kernels
  do j = jts, jte
    do 330 i = its, ite
      a(i,j) = a(i,j) + 1.0
330 continue
    if (force) then
      if (flux .eq. 1) goto 350
    endif
    do i = its, ite
      qfx(i,j) = 0.
    enddo
350 continue
  enddo
  !$acc end kernels
end subroutine

! CHECK-LABEL: func.func @_QPkernels_goto_over_inner
! CHECK:         acc.kernels {
! CHECK:           acc.loop private({{.*}}) control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! CHECK:             scf.execute_region no_inline {
! The inner loops are untouched by the wrap around the outer body.
! CHECK:               acc.loop private({{.*}}) control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! CHECK:                 acc.yield
! The GOTO and the branches it feeds stay inside the region.
! CHECK:               cf.cond_br %{{[0-9]+}}, ^bb[[THEN:[0-9]+]], ^bb[[SKIP:[0-9]+]]
! CHECK:               acc.loop private({{.*}}) control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! CHECK:                 acc.yield
! CHECK:               scf.yield
! CHECK:             acc.yield

! A CYCLE targets the EndDoStmt, the boundary between the loop body and the
! loop control, so inside the wrap it leaves the region at its yield.
subroutine parallel_loop_cycle(a, n)
  real :: a(n)
  !$acc parallel loop
  do i = 1, n
    if (a(i) > 0.0) then
      a(i) = 1.0
      cycle
    end if
    a(i) = 2.0
  end do
end subroutine

! CHECK-LABEL: func.func @_QPparallel_loop_cycle
! CHECK:         acc.parallel combined(loop) {
! CHECK:           acc.loop private({{.*}}) control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32) {
! CHECK:             scf.execute_region no_inline {
! CHECK:               cf.cond_br %{{[0-9]+}}, ^bb[[CYCLE:[0-9]+]], ^bb[[BODY:[0-9]+]]
! CHECK:             ^bb[[CYCLE]]:
! CHECK:               cf.br ^bb[[EXIT:[0-9]+]]
! CHECK:             ^bb[[BODY]]:
! CHECK:               cf.br ^bb[[EXIT]]
! CHECK:             ^bb[[EXIT]]:
! CHECK:               scf.yield
! CHECK:             acc.yield
