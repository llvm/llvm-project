! Test lowering of `lastprivate(conditional:)` on sections in a target region.
!
! Case 1: target + parallel + sections
! Case 2: target teams + parallel sections (per-team struct inside omp.teams)
! Case 3: target + orphaned sections (no explicit parallel)
! Case 4: firstprivate + conditional lastprivate on the same item
! Case 5: outlined orphaned sections in a declare-target routine
!
! In the lexically-nested cases (1-4) the struct is a per-team/target alloca.
! Case 5 has no lexical enclosing target/parallel, so it uses the module-scope
! global plus the omp_get_level/llvm.trap device-safe nesting guard.

! RUN: bbc -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s

! No host-only runtime abort should be emitted for the device (the orphaned
! Case 5 aborts with llvm.trap, not _FortranAStopStatementText).
! CHECK-NOT: _FortranAStopStatementText

! =============================================================================
! Case 1: target + parallel + sections
! =============================================================================
! CHECK-LABEL: func @_QPtarget_sections

subroutine target_sections(x, y)
  implicit none
  integer :: x, y

  !$omp target map(tofrom: x, y)
  !$omp parallel
  !$omp sections lastprivate(conditional: x, y)
  !$omp section
    x = 11
  !$omp section
    y = 22
  !$omp end sections
  !$omp end parallel
  !$omp end target
end subroutine

! -- Per-target struct alloca at the start of the target body -----------------
! CHECK:         omp.target
! CHECK:           %[[STRUCT:.*]] = fir.alloca !fir.type<{{.*}}{x:i32,y:i32,$x:i64,$y:i64}> {pinned}
! CHECK:           fir.coordinate_of %[[STRUCT]], x
! CHECK:           fir.coordinate_of %[[STRUCT]], y
! CHECK:           omp.parallel {
! CHECK:             omp.sections
! CHECK-SAME:          reduction(byref @lp_cond_byref_rec__lp_cond_t
! -- Copy-back inside the parallel, wrapped in omp.single, guarded by idx >= 0 -
! CHECK:             omp.single {
! CHECK:               fir.coordinate_of %[[STRUCT]], x
! CHECK:               arith.cmpi sge, %{{.*}}, %{{.*}} : i64
! CHECK:               fir.if
! CHECK:                 fir.store
! CHECK:               fir.coordinate_of %[[STRUCT]], y
! CHECK:               arith.cmpi sge, %{{.*}}, %{{.*}} : i64
! CHECK:               fir.if
! CHECK:                 fir.store
! CHECK:               omp.terminator

! =============================================================================
! Case 2: target teams + parallel sections (struct inside omp.teams)
! =============================================================================
! CHECK-LABEL: func @_QPtarget_teams_sections

subroutine target_teams_sections(x, y)
  implicit none
  integer :: x, y

  !$omp target teams map(tofrom: x, y)
  !$omp parallel sections lastprivate(conditional: x, y)
  !$omp section
    x = 11
  !$omp section
    y = 22
  !$omp end parallel sections
  !$omp end target teams
end subroutine

! -- Struct alloca inside the omp.teams body (per team) -----------------------
! CHECK:         omp.target
! CHECK:           omp.teams {
! CHECK:             %[[STRUCT2:.*]] = fir.alloca !fir.type<{{.*}}{x:i32,y:i32,$x:i64,$y:i64}> {pinned}
! CHECK:             omp.parallel {
! CHECK:               omp.sections
! CHECK-SAME:            reduction(byref @lp_cond_byref_rec__lp_cond_t
! -- Copy-back inside the parallel, wrapped in omp.single ---------------------
! CHECK:               omp.single {
! CHECK:                 arith.cmpi sge, %{{.*}}, %{{.*}} : i64
! CHECK:                 fir.if
! CHECK:                 arith.cmpi sge, %{{.*}}, %{{.*}} : i64
! CHECK:                 fir.if

! =============================================================================
! Case 3: target + orphaned sections (no explicit parallel)
! =============================================================================
! The sections is directly in the target body (bound to the target's single
! initial thread).
! CHECK-LABEL: func @_QPtarget_orphaned_sections

subroutine target_orphaned_sections(x, y)
  implicit none
  integer :: x, y

  !$omp target map(tofrom: x, y)
  !$omp sections lastprivate(conditional: x, y)
  !$omp section
    x = 11
  !$omp section
    y = 22
  !$omp end sections
  !$omp end target
end subroutine

! -- Struct alloca in the target body; sections directly in target (no parallel)
! CHECK:         omp.target
! CHECK:           %[[STRUCT3:.*]] = fir.alloca !fir.type<{{.*}}{x:i32,y:i32,$x:i64,$y:i64}> {pinned}
! CHECK-NOT:       omp.parallel
! CHECK:           omp.sections
! CHECK-SAME:        reduction(byref @lp_cond_byref_rec__lp_cond_t
! -- Guarded copy-back after the sections ------------------------------------
! CHECK:           fir.coordinate_of %[[STRUCT3]], x
! CHECK:           arith.cmpi sge, %{{.*}}, %{{.*}} : i64
! CHECK:           fir.if
! CHECK:             fir.store
! CHECK:           omp.terminator

! =============================================================================
! Case 4: firstprivate + conditional lastprivate on the same item
! =============================================================================
! The list item is bound to the struct value field, so firstprivate must seed
! that field with the incoming value.
! CHECK-LABEL: func @_QPtarget_sections_fp

subroutine target_sections_fp(x)
  implicit none
  integer :: x

  !$omp target map(tofrom: x)
  !$omp parallel
  !$omp sections firstprivate(x) lastprivate(conditional: x)
  !$omp section
    x = x + 1
  !$omp end sections
  !$omp end parallel
  !$omp end target
end subroutine

! CHECK:         omp.target
! -- The struct alloca is hoisted to the top of the target region; the mapped
! -- variable is declared, and the firstprivate seed loads it AFTER the declare
! -- (placing the seed after the declares is what avoids the earlier "does not
! -- dominate" verifier error).
! CHECK:           %[[STRUCT4:.*]] = fir.alloca !fir.type<{{.*}}{x:i32,$x:i64}> {pinned}
! CHECK:           %[[XDECL:.*]]:2 = hlfir.declare %{{.*}}Ex"
! -- default init writes the value field once ...
! CHECK:           fir.coordinate_of %[[STRUCT4]], x
! CHECK:           fir.coordinate_of %[[STRUCT4]], $x
! -- ... then the firstprivate seed writes it again from the loaded x ---------
! CHECK:           fir.coordinate_of %[[STRUCT4]], x
! CHECK:           %[[XVAL:.*]] = fir.load %[[XDECL]]#0
! CHECK:           fir.store %[[XVAL]]
! CHECK:           omp.parallel {
! CHECK:             omp.sections
! CHECK-SAME:          reduction(byref @lp_cond_byref_rec__lp_cond_t

! =============================================================================
! Case 5: OUTLINED orphaned sections in a declare-target routine
! =============================================================================
! No lexical enclosing target/parallel, so the struct is the module-scope global
! (shared by all threads that call the routine), guarded by omp_get_level /
! llvm.trap (device-safe abort).  At runtime this is called from target+parallel.
module condlp_sec_mod
contains
  subroutine process(x, y)
    !$omp declare target
    integer, intent(inout) :: x, y
    !$omp sections lastprivate(conditional: x, y)
    !$omp section
      x = 11
    !$omp section
      y = 22
    !$omp end sections
  end subroutine
end module

! CHECK-LABEL: func.func @_QMcondlp_sec_modPprocess
! CHECK-SAME:    attributes {omp.declare_target =
! -- Nesting guard: omp_get_level (C name), then llvm.trap ---------------------
! CHECK:         %[[LEVEL:.*]] = fir.call @omp_get_level() {{.*}} : () -> i32
! CHECK:         arith.cmpi sgt, %[[LEVEL]], %{{.*}} : i32
! CHECK:         fir.if
! CHECK:           fir.call @llvm.trap()
! -- Uses the module-scope global (not an alloca) -----------------------------
! CHECK:         %[[GADDR:.*]] = fir.address_of(@_lp_cond_global.{{l[0-9]+\.[0-9]+}})
! -- Init inside omp.single ---------------------------------------------------
! CHECK:         omp.single {
! CHECK:           fir.coordinate_of %[[GADDR]], x
! CHECK:           fir.coordinate_of %[[GADDR]], y
! CHECK:           omp.terminator
! -- sections with reduction on the global struct -----------------------------
! CHECK:         omp.sections
! CHECK-SAME:      reduction(byref @lp_cond_byref_rec__lp_cond_t.{{l[0-9]+\.[0-9]+}} %[[GADDR]]
! -- Copy-back in omp.single, one guarded store per item ----------------------
! CHECK:         omp.single {
! CHECK:           arith.cmpi sge, %{{.*}}, %{{.*}} : i64
! CHECK:           fir.if
! CHECK:           arith.cmpi sge, %{{.*}}, %{{.*}} : i64
! CHECK:           fir.if
! CHECK:           omp.terminator
! -- Module-scope global for the struct ---------------------------------------
! CHECK:       fir.global internal @_lp_cond_global.{{l[0-9]+\.[0-9]+}} : !fir.type<_lp_cond_t.{{l[0-9]+\.[0-9]+}}{x:i32,y:i32,$x:i64,$y:i64}>
