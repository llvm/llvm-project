! Test lowering of `lastprivate(conditional:)` on worksharing loops in a target
! region.
!
! Case 1: target parallel do (combined)
! Case 2: target + separate parallel / do
! Case 3: standalone target do (no enclosing parallel)
! Case 4: target teams (per-team struct placed inside omp.teams)
! Case 5: non-unit positive step (host_eval step is a positive constant)
! Case 6: multiple list items of different types (real + logical)
! Case 7: firstprivate + conditional lastprivate on a target loop
! Case 8: outlined orphaned do in a declare-target routine
!
! Cases 1-7 place the conditional-LP struct (fir.alloca) at the start of the
! omp.target body -- or the omp.teams body when a teams is present, so each team
! gets its own copy.  In cases 1, 2 and 4 the copy-back is placed after
! omp.parallel; in case 3 there is no parallel, so it goes right after the
! wsloop.  Case 8 is orphaned (no lexical target/parallel) and uses the
! module-scope global plus the omp_get_level/llvm.trap device-safe guard.

! RUN: bbc -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s

! -- declare_reduction for the struct type ------------------------------------
! The combiner keeps the value from the higher canonical index (sgt on i64).
! The per-case struct layouts are checked at the func level below; declare
! reductions are emitted in reverse source order, so don't pin the field types
! here.
! CHECK-LABEL: omp.declare_reduction @lp_cond_byref_rec__lp_cond_t
! CHECK-SAME:    : !fir.ref<!fir.type<_lp_cond_t.{{.*}}>>
! CHECK:       } combiner {
! CHECK:         arith.cmpi sgt, %{{.*}}, %{{.*}} : i64
! CHECK:       }

! =============================================================================
! Case 1: target parallel do (combined)
! =============================================================================
! CHECK-LABEL: func @_QPtarget_parallel_do

subroutine target_parallel_do(n, a)
  implicit none
  integer, intent(in) :: n
  integer, intent(inout) :: a(n)
  integer :: x, i

  x = 0
  !$omp target parallel do map(tofrom: x) map(to: a) lastprivate(conditional: x)
  do i = 1, n
    if (a(i) > 0) then
      x = a(i)
    end if
  end do
  !$omp end target parallel do
end subroutine

! -- Struct alloca at beginning of omp.target body ----------------------------
! CHECK:         omp.target
! CHECK:           %[[STRUCT:.*]] = fir.alloca !fir.type<_lp_cond_t.{{l[0-9]+\.[0-9]+}}{x:i32,$x:i64}> {pinned}

! -- Init: x=0, $x=-1 --------------------------------------------------------
! CHECK:           %[[XCOORD:.*]] = fir.coordinate_of %[[STRUCT]], x
! CHECK:           %[[C0:.*]] = arith.constant 0 : i32
! CHECK:           fir.store %[[C0]] to %[[XCOORD]]
! CHECK:           %[[KCOORD:.*]] = fir.coordinate_of %[[STRUCT]], $x
! CHECK:           %[[CM1:.*]] = arith.constant -1 : i64
! CHECK:           fir.store %[[CM1]] to %[[KCOORD]]

! -- omp.parallel with wsloop reduction on the struct -------------------------
! CHECK:           omp.parallel {
! CHECK:             omp.wsloop
! CHECK-SAME:          reduction(byref @lp_cond_byref_rec__lp_cond_t

! -- Index + guarded commit inside the loop body ------------------------------
! CHECK:               omp.loop_nest
! CHECK:                 %[[IV:.*]] = fir.convert %{{.*}} : (i32) -> i64
! CHECK:                 fir.coordinate_of %{{.*}}, $x
! CHECK:                 arith.cmpi sge, %[[IV]], %{{.*}} : i64
! CHECK:                 fir.if
! CHECK:                   fir.store

! -- Copy-back after parallel, guarded by idx >= 0 ----------------------------
! CHECK:           %[[XVAL:.*]] = fir.load
! CHECK:           %[[KVAL:.*]] = fir.load
! CHECK:           %[[C0_I64:.*]] = arith.constant 0 : i64
! CHECK:           %[[CMP:.*]] = arith.cmpi sge, %[[KVAL]], %[[C0_I64]] : i64
! CHECK:           fir.if %[[CMP]] {
! CHECK:             fir.store %[[XVAL]] to %{{.*}}
! CHECK:           }
! CHECK:           omp.terminator

! =============================================================================
! Case 2: target + separate parallel / do
! =============================================================================
! CHECK-LABEL: func @_QPtarget_separate_parallel_do

subroutine target_separate_parallel_do(n, a)
  implicit none
  integer, intent(in) :: n
  integer, intent(inout) :: a(n)
  integer :: x, i

  x = 0
  !$omp target map(tofrom: x) map(to: a)
  !$omp parallel
  !$omp do lastprivate(conditional: x)
  do i = 1, n
    if (a(i) > 0) then
      x = a(i)
    end if
  end do
  !$omp end do
  !$omp end parallel
  !$omp end target
end subroutine

! -- Same structure: alloca at target body, reduction on wsloop, guarded copy-back
! CHECK:         omp.target
! CHECK:           %[[STRUCT2:.*]] = fir.alloca !fir.type<_lp_cond_t.{{l[0-9]+\.[0-9]+}}{x:i32,$x:i64}> {pinned}
! CHECK:           fir.coordinate_of %[[STRUCT2]], x
! CHECK:           fir.coordinate_of %[[STRUCT2]], $x
! CHECK:           omp.parallel {
! CHECK:             omp.wsloop
! CHECK-SAME:          reduction(byref @lp_cond_byref_rec__lp_cond_t
! CHECK:           arith.cmpi sge, %{{.*}}, %{{.*}} : i64
! CHECK:           fir.if
! CHECK:           omp.terminator

! =============================================================================
! Case 3: standalone target do (no enclosing parallel)
! =============================================================================
! wsloop directly inside omp.target with no omp.parallel: it runs on the
! target's single initial thread, so copy-back goes right after the wsloop with
! no barrier/single.
! CHECK-LABEL: func @_QPtarget_standalone_do

subroutine target_standalone_do(n, a)
  implicit none
  integer, intent(in) :: n
  integer, intent(inout) :: a(n)
  integer :: x, i

  x = 0
  !$omp target map(tofrom: x) map(to: a)
  !$omp do lastprivate(conditional: x)
  do i = 1, n
    if (a(i) > 0) then
      x = a(i)
    end if
  end do
  !$omp end do
  !$omp end target
end subroutine

! -- Struct alloca at beginning of omp.target body ----------------------------
! CHECK:         omp.target
! CHECK:           %[[STRUCT3:.*]] = fir.alloca !fir.type<_lp_cond_t.{{l[0-9]+\.[0-9]+}}{x:i32,$x:i64}> {pinned}
! CHECK:           fir.coordinate_of %[[STRUCT3]], x
! CHECK:           fir.coordinate_of %[[STRUCT3]], $x
! -- No enclosing parallel: the wsloop is directly in the target body ---------
! CHECK-NOT:       omp.parallel
! CHECK:           omp.wsloop
! CHECK-SAME:        reduction(byref @lp_cond_byref_rec__lp_cond_t
! -- Canonical index is (iv - lb)/step (generic kernel, not host_eval) --------
! CHECK:             omp.loop_nest
! CHECK:               arith.subi
! CHECK:               arith.divsi
! -- Copy-back right after the wsloop, guarded by idx >= 0 --------------------
! CHECK:           arith.cmpi sge, %{{.*}}, %{{.*}} : i64
! CHECK:           fir.if
! CHECK:             fir.store
! CHECK:           omp.terminator

! =============================================================================
! Case 4: target teams (per-team struct placed inside omp.teams)
! =============================================================================
! With an enclosing omp.teams the struct alloca is placed at the start of the
! teams body (not the target body) so every team reduces into its own copy.
! The copy-back runs after omp.parallel but still inside the teams region.
! CHECK-LABEL: func @_QPtarget_teams_do

subroutine target_teams_do(n, x)
  implicit none
  integer, intent(in) :: n
  integer :: x, i

  !$omp target teams
  !$omp parallel do lastprivate(conditional: x)
  do i = 1, n
    if (mod(i, 3) == 0) x = i
  end do
  !$omp end parallel do
  !$omp end target teams
end subroutine

! CHECK:         omp.target
! CHECK:           omp.teams {
! -- Per-team struct alloca inside the teams body -----------------------------
! CHECK:             %[[STRUCT4:.*]] = fir.alloca !fir.type<_lp_cond_t.{{l[0-9]+\.[0-9]+}}{x:i32,$x:i64}> {pinned}
! CHECK:             fir.coordinate_of %[[STRUCT4]], x
! CHECK:             fir.coordinate_of %[[STRUCT4]], $x
! CHECK:             omp.parallel {
! CHECK:               omp.wsloop
! CHECK-SAME:            reduction(byref @lp_cond_byref_rec__lp_cond_t
! -- Copy-back after the parallel, still inside the teams region --------------
! CHECK:             arith.cmpi sge, %{{.*}}, %{{.*}} : i64
! CHECK:             fir.if
! CHECK:           omp.terminator

! =============================================================================
! Case 5: non-unit positive step
! =============================================================================
! The host_eval step is the positive constant 2, so the raw-IV fallback is
! still valid (single forward loop, lb >= 0, step > 0) and the index is the
! raw loop IV, not (iv - lb)/step.
! CHECK-LABEL: func @_QPtarget_step2

subroutine target_step2(n, a)
  implicit none
  integer, intent(in) :: n
  integer, intent(inout) :: a(n)
  integer :: x, i

  x = 0
  !$omp target parallel do map(tofrom: x) map(to: a) lastprivate(conditional: x)
  do i = 1, n, 2
    if (a(i) > 0) x = a(i)
  end do
  !$omp end target parallel do
end subroutine

! CHECK:         omp.target
! CHECK-SAME:      kernel_type(spmd)
! CHECK:           %[[STRUCT5:.*]] = fir.alloca !fir.type<_lp_cond_t.{{l[0-9]+\.[0-9]+}}{x:i32,$x:i64}> {pinned}
! CHECK:           omp.parallel {
! CHECK:             omp.wsloop
! CHECK-SAME:          reduction(byref @lp_cond_byref_rec__lp_cond_t
! -- host_eval bounds: the index is the raw IV (no subi/divsi on the IV) ------
! CHECK:               omp.loop_nest
! CHECK:                 %[[IV5:.*]] = fir.convert %{{.*}} : (i32) -> i64
! CHECK:                 arith.cmpi sge, %[[IV5]], %{{.*}} : i64
! CHECK:           arith.cmpi sge, %{{.*}}, %{{.*}} : i64
! CHECK:           omp.terminator

! =============================================================================
! Case 6: multiple list items of different types (real + logical)
! =============================================================================
! The struct packs both value fields (r:f32, l:logical<4>) and both index
! fields ($r, $l).  Each list item gets its own guarded commit in the loop and
! its own guarded copy-back after the parallel.
! CHECK-LABEL: func @_QPtarget_real_logical

subroutine target_real_logical(n, a, r, l)
  implicit none
  integer, intent(in) :: n
  integer, intent(in) :: a(n)
  real :: r
  logical :: l
  integer :: i

  !$omp target parallel do map(tofrom: r, l) map(to: a) lastprivate(conditional: r, l)
  do i = 1, n
    if (a(i) > 0) then
      r = real(a(i))
      l = .true.
    end if
  end do
  !$omp end target parallel do
end subroutine

! -- Struct packs r:f32, l:logical<4> plus $r,$l index fields -----------------
! CHECK:         omp.target
! CHECK:           %[[STRUCT6:.*]] = fir.alloca !fir.type<{{.*}}{r:f32,l:!fir.logical<4>,$r:i64,$l:i64}> {pinned}
! CHECK:           fir.coordinate_of %[[STRUCT6]], r
! CHECK:           fir.coordinate_of %[[STRUCT6]], l
! CHECK:           fir.coordinate_of %[[STRUCT6]], $r
! CHECK:           fir.coordinate_of %[[STRUCT6]], $l
! CHECK:           omp.parallel {
! CHECK:             omp.wsloop
! CHECK-SAME:          reduction(byref @lp_cond_byref_rec__lp_cond_t
! -- Two guarded commits in the loop, one per list item -----------------------
! CHECK:               arith.cmpi sge, %{{.*}}, %{{.*}} : i64
! CHECK:               arith.cmpi sge, %{{.*}}, %{{.*}} : i64
! -- Two guarded copy-backs after the parallel --------------------------------
! CHECK:           fir.coordinate_of %[[STRUCT6]], r
! CHECK:           arith.cmpi sge, %{{.*}}, %{{.*}} : i64
! CHECK:           fir.coordinate_of %[[STRUCT6]], l
! CHECK:           arith.cmpi sge, %{{.*}}, %{{.*}} : i64
! CHECK:           omp.terminator

! =============================================================================
! Case 7: firstprivate + conditional lastprivate on a target loop
! =============================================================================
! The same list item is firstprivate (seeded from the incoming value) and
! conditional lastprivate.  The loop body reads the firstprivate copy
! (x = x + i), then commits it to the struct value field under the index guard.
! CHECK-LABEL: func @_QPtarget_do_firstprivate

subroutine target_do_firstprivate(n, x)
  implicit none
  integer, intent(in) :: n
  integer :: x, i
  x = 100
  !$omp target parallel do map(tofrom: x) firstprivate(x) lastprivate(conditional: x)
  do i = 1, n
    if (i == 7) x = x + i
  end do
  !$omp end target parallel do
end subroutine

! CHECK:         omp.target
! CHECK:           %[[STRUCT7:.*]] = fir.alloca !fir.type<_lp_cond_t.{{l[0-9]+\.[0-9]+}}{x:i32,$x:i64}> {pinned}
! CHECK:           omp.parallel {
! -- x is BOTH firstprivate (private copy) and carries the conditional reduction
! CHECK:             omp.wsloop
! CHECK-SAME:          private(@{{.*}}Ex_firstprivate
! CHECK-SAME:          reduction(byref @lp_cond_byref_rec__lp_cond_t
! -- Loop body reads the firstprivate copy, then a guarded commit to the struct
! CHECK:               omp.loop_nest
! CHECK:                 arith.addi
! CHECK:                 fir.coordinate_of %{{.*}}, $x
! CHECK:                 arith.cmpi sge, %{{.*}}, %{{.*}} : i64
! CHECK:                 fir.if
! CHECK:                   fir.store
! -- Guarded copy-back after the parallel -------------------------------------
! CHECK:           fir.coordinate_of %[[STRUCT7]], x
! CHECK:           arith.cmpi sge, %{{.*}}, %{{.*}} : i64
! CHECK:           fir.if
! CHECK:             fir.store
! CHECK:           omp.terminator

! =============================================================================
! Case 8: OUTLINED orphaned do in a declare-target routine
! =============================================================================
! No lexical enclosing target/parallel, so the struct is the module-scope global
! (shared by all threads that call the routine), guarded by omp_get_level /
! llvm.trap (device-safe abort).  At runtime this is called from target+parallel.
module condlp_mod
contains
  subroutine process(n, a, x)
    !$omp declare target
    integer, intent(in) :: n, a(n)
    integer, intent(inout) :: x
    integer :: i
    !$omp do lastprivate(conditional: x)
    do i = 1, n
      if (a(i) > 0) x = a(i)
    end do
    !$omp end do
  end subroutine
end module

! CHECK-LABEL: func.func @_QMcondlp_modPprocess
! CHECK-SAME:    attributes {omp.declare_target =
! -- Nesting guard: omp_get_level (C name), then llvm.trap ---------------------
! CHECK:         %[[LEVEL:.*]] = fir.call @omp_get_level() {{.*}} : () -> i32
! CHECK:         arith.cmpi sgt, %[[LEVEL]], %{{.*}} : i32
! CHECK:         fir.if
! CHECK:           fir.call @llvm.trap()
! -- Uses the module-scope global (not an alloca) -----------------------------
! CHECK-NOT:     fir.alloca !fir.type<_lp_cond_t
! CHECK:         %[[GADDR:.*]] = fir.address_of(@_lp_cond_global.{{l[0-9]+\.[0-9]+}})
! -- Init inside omp.single ---------------------------------------------------
! CHECK:         omp.single {
! CHECK:           fir.coordinate_of %[[GADDR]], x
! CHECK:           fir.coordinate_of %[[GADDR]], $x
! CHECK:           omp.terminator
! -- wsloop with reduction on the global struct -------------------------------
! CHECK:         omp.wsloop
! CHECK-SAME:      reduction(byref @lp_cond_byref_rec__lp_cond_t.{{l[0-9]+\.[0-9]+}} %[[GADDR]]
! -- Copy-back in omp.single with idx >= 0 guard ------------------------------
! CHECK:         omp.single {
! CHECK:           arith.cmpi sge, %{{.*}}, %{{.*}} : i64
! CHECK:           fir.if
! CHECK:           omp.terminator
! -- Module-scope global for the struct ---------------------------------------
! CHECK:       fir.global internal @_lp_cond_global.{{l[0-9]+\.[0-9]+}} : !fir.type<_lp_cond_t.{{l[0-9]+\.[0-9]+}}{x:i32,$x:i64}>
