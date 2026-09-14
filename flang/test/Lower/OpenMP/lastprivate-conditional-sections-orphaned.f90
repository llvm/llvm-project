! Test lowering of `lastprivate(conditional:)` on an ORPHANED omp sections
! construct (sections inside a subroutine called from a parallel region).
!
! Because the subroutine has no enclosing omp.parallel, a stack alloca would
! give each thread a private copy and the cross-thread reduction would never
! merge.  The lowering must therefore place the reduction struct in a
! module-scope fir.global internal rather than a fir.alloca.

! RUN: %flang_fc1 -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s

subroutine test_orphaned_sections(n)
  implicit none
  integer, intent(inout) :: n

  !$omp sections lastprivate(conditional: n)
    !$omp section
    n = 10
    !$omp section
    n = 20
  !$omp end sections
end subroutine

! -- declare_reduction for the struct type ------------------------------------
! CHECK-LABEL: omp.declare_reduction @lp_cond_byref_rec__lp_cond_t
! CHECK-SAME:    : !fir.ref<!fir.type<_lp_cond_t.{{l[0-9]+\.[0-9]+}}{n:i32,$n:i64}>>

! -- Function body: address_of global (no fir.alloca for the struct) ----------
! CHECK-LABEL: func.func @_QPtest_orphaned_sections

! -- Runtime guard: abort if called from nested parallelism ------------------
! Guard is emitted BEFORE init to avoid racing on the global.
! CHECK:         %[[LEVEL:.*]] = fir.call @omp_get_level_() {{.*}} : () -> i32
! CHECK:         %[[ONE:.*]] = arith.constant 1 : i32
! CHECK:         %[[NESTED:.*]] = arith.cmpi sgt, %[[LEVEL]], %[[ONE]] : i32
! CHECK:         fir.if %[[NESTED]] {
! CHECK:           fir.call @_FortranAStopStatementText
! CHECK:         }

! CHECK-NOT:     fir.alloca !fir.type<_lp_cond_t
! CHECK:         %[[GADDR:.*]] = fir.address_of(@_lp_cond_global.{{l[0-9]+\.[0-9]+}}) : !fir.ref<!fir.type<_lp_cond_t.{{l[0-9]+\.[0-9]+}}{n:i32,$n:i64}>>

! -- Init sentinels written inside omp.single --------------------------------
! CHECK:         omp.single {
! CHECK:           %[[NCOORD:.*]] = fir.coordinate_of %[[GADDR]], n
! CHECK:           %[[C0:.*]] = arith.constant 0 : i32
! CHECK:           fir.store %[[C0]] to %[[NCOORD]]
! CHECK:           %[[KNCOORD:.*]] = fir.coordinate_of %[[GADDR]], $n
! CHECK:           %[[CM1:.*]] = arith.constant -1 : i64
! CHECK:           fir.store %[[CM1]] to %[[KNCOORD]]
! CHECK:           omp.terminator
! CHECK:         }

! -- Sections carries the global address as a by-ref reduction ---------------
! CHECK:         omp.sections
! CHECK-SAME:      reduction(byref @lp_cond_byref_rec__lp_cond_t
! CHECK-SAME:        %[[GADDR]]

! -- Section 0: index constant hoisted to entry, stored after assignment ------
! CHECK:           omp.section {
! CHECK:             %[[IDX0:.*]] = arith.constant 0 : i64
! CHECK:             hlfir.assign
! CHECK:             fir.store %[[IDX0]]

! -- Section 1: index constant hoisted to entry, stored after assignment ------
! CHECK:           omp.section {
! CHECK:             %[[IDX1:.*]] = arith.constant 1 : i64
! CHECK:             hlfir.assign
! CHECK:             fir.store %[[IDX1]]

! -- Copy-back: load winning value from global and store to dummy arg ---------
! CHECK:         fir.coordinate_of %[[GADDR]], n
! CHECK:         fir.load
! CHECK:         fir.coordinate_of %[[GADDR]], $n
! CHECK:         fir.load
! CHECK:         arith.cmpi sge
! CHECK:         fir.if
! CHECK:           fir.store
! CHECK:         }

! -- Module-level global declared at end of module (not a stack alloca) -------
! CHECK: fir.global internal @_lp_cond_global.{{l[0-9]+\.[0-9]+}} : !fir.type<_lp_cond_t.{{l[0-9]+\.[0-9]+}}{n:i32,$n:i64}>
