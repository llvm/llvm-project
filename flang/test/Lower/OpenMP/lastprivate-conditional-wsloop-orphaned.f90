! Test lowering of `lastprivate(conditional:)` on an ORPHANED omp do loop
! (wsloop inside a subroutine called from a parallel region).
!
! Because the subroutine has no enclosing omp.parallel, a stack alloca would
! give each thread a private copy and the cross-thread reduction would never
! merge.  The lowering must therefore place the reduction struct in a
! module-scope fir.global internal rather than a fir.alloca.

! RUN: %flang_fc1 -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s

subroutine test_orphaned_wsloop(n, x)
  implicit none
  integer, intent(in) :: n
  integer, intent(inout) :: x
  integer :: k

  !$omp do lastprivate(conditional: x)
  do k = 1, n
    x = k
  end do
  !$omp end do
end subroutine

! -- declare_reduction for the struct type ------------------------------------
! CHECK-LABEL: omp.declare_reduction @lp_cond_byref_rec__lp_cond_t
! CHECK-SAME:    : !fir.ref<!fir.type<_lp_cond_t.{{l[0-9]+\.[0-9]+}}{x:i32,$x:i64}>>

! -- Function body: address_of global (no fir.alloca for the struct) ----------
! CHECK-LABEL: func.func @_QPtest_orphaned_wsloop

! -- Runtime guard: abort if called from nested parallelism ------------------
! Guard is emitted BEFORE init to avoid racing on the global.
! CHECK:         %[[LEVEL:.*]] = fir.call @omp_get_level_() {{.*}} : () -> i32
! CHECK:         %[[ONE:.*]] = arith.constant 1 : i32
! CHECK:         %[[NESTED:.*]] = arith.cmpi sgt, %[[LEVEL]], %[[ONE]] : i32
! CHECK:         fir.if %[[NESTED]] {
! CHECK:           fir.call @_FortranAStopStatementText
! CHECK:         }

! CHECK-NOT:     fir.alloca !fir.type<_lp_cond_t
! CHECK:         %[[GADDR:.*]] = fir.address_of(@_lp_cond_global.{{l[0-9]+\.[0-9]+}}) : !fir.ref<!fir.type<_lp_cond_t.{{l[0-9]+\.[0-9]+}}{x:i32,$x:i64}>>

! -- Init sentinels written inside omp.single --------------------------------
! CHECK:         omp.single {
! CHECK:           %[[XCOORD:.*]] = fir.coordinate_of %[[GADDR]], x
! CHECK:           %[[C0:.*]] = arith.constant 0 : i32
! CHECK:           fir.store %[[C0]] to %[[XCOORD]]
! CHECK:           %[[KXCOORD:.*]] = fir.coordinate_of %[[GADDR]], $x
! CHECK:           %[[CM1:.*]] = arith.constant -1 : i64
! CHECK:           fir.store %[[CM1]] to %[[KXCOORD]]
! CHECK:           omp.terminator
! CHECK:         }

! -- Wsloop carries the global address as a by-ref reduction -----------------
! CHECK:         omp.wsloop
! CHECK-SAME:      reduction(byref @lp_cond_byref_rec__lp_cond_t
! CHECK-SAME:        %[[GADDR]]
! CHECK-SAME:        -> %[[SARG:.*]] :

! -- Loop body: struct value and index fields updated -------------------------
! CHECK:           omp.loop_nest
! CHECK:             fir.coordinate_of %[[SARG]], x
! CHECK:             hlfir.assign
! CHECK:             fir.coordinate_of %[[SARG]], $x
! CHECK:             fir.store

! -- Copy-back: load winning value from global and store to dummy arg ---------
! CHECK:         fir.coordinate_of %[[GADDR]], x
! CHECK:         fir.load
! CHECK:         fir.coordinate_of %[[GADDR]], $x
! CHECK:         fir.load
! CHECK:         arith.cmpi sge
! CHECK:         fir.if
! CHECK:           fir.store
! CHECK:         }

! -- Module-level global declared at end of module (not a stack alloca) -------
! CHECK: fir.global internal @_lp_cond_global.{{l[0-9]+\.[0-9]+}} : !fir.type<_lp_cond_t.{{l[0-9]+\.[0-9]+}}{x:i32,$x:i64}>
