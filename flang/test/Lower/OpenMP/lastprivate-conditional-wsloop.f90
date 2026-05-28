! Test lowering of `lastprivate(conditional:)` on a worksharing do loop
! with multiple variables.  The lowering must:
!   1. Build a packed struct type {val, val, ..., idx, idx, ...}
!   2. Create an omp.declare_reduction with identity 0 / -1
!   3. Inject the struct as a by-ref reduction variable on the wsloop
!   4. Rewrite assignments to use struct value fields + store canonical IV
!   5. Copy back the winning values after the wsloop

! RUN: bbc -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s

subroutine test_conditional_lp(n, a, x, y)
  implicit none
  integer, intent(in) :: n
  integer, intent(in) :: a(n)
  integer, intent(inout) :: x, y
  integer :: k

  !$omp parallel do lastprivate(conditional: x, y)
  do k = 1, n
    if (a(k) < 150) then
      x = k + 1
    end if
    if (a(k) < 100) then
      y = k
    end if
  end do
  !$omp end parallel do
end subroutine

! -- declare_reduction with struct type containing value/index pairs ----------
! CHECK-LABEL: omp.declare_reduction @lp_cond_byref_rec__lp_cond_t
! CHECK-SAME:    : !fir.ref<!fir.type<_lp_cond_t.{{l[0-9]+\.[0-9]+}}{x:i32,y:i32,$x:i64,$y:i64}>>

! -- Init region: value fields = 0, index fields = -1 ------------------------
! CHECK:       init {
! CHECK-DAG:     arith.constant 0 : i32
! CHECK-DAG:     arith.constant -1 : i64
! CHECK:       }

! -- Combiner: sgt on i64 index fields, two pairs ----------------------------
! CHECK:       combiner {
! CHECK:         arith.cmpi sgt, %{{.*}}, %{{.*}} : i64
! CHECK:         fir.if
! CHECK:         arith.cmpi sgt, %{{.*}}, %{{.*}} : i64
! CHECK:         fir.if
! CHECK:         omp.yield

! -- Struct alloca + init (0 / -1) before parallel ----------------------------
! CHECK-LABEL: func.func @_QPtest_conditional_lp
! CHECK:         %[[STRUCT:.*]] = fir.alloca !fir.type<_lp_cond_t.{{l[0-9]+\.[0-9]+}}{x:i32,y:i32,$x:i64,$y:i64}>
! CHECK-DAG:     arith.constant 0 : i32
! CHECK-DAG:     arith.constant -1 : i64

! -- Wsloop carries the struct as a by-ref reduction --------------------------
! CHECK:         omp.parallel {
! CHECK:           omp.wsloop
! CHECK-SAME:        reduction(byref @lp_cond_byref_rec__lp_cond_t
! CHECK-SAME:          %[[STRUCT]]
! CHECK-SAME:          -> %[[SARG:.*]] :

! -- Loop body: struct value fields used, canonical IV stored to index fields -
! CHECK:             omp.loop_nest (%{{.*}}) : i32
! CHECK-DAG:           fir.coordinate_of %[[SARG]], x
! CHECK-DAG:           fir.coordinate_of %[[SARG]], y
! CHECK:               fir.if
! CHECK:                 hlfir.assign
! CHECK:                 fir.coordinate_of %[[SARG]], $x
! CHECK:                 fir.store %{{.*}} : !fir.ref<i64>
! CHECK:               fir.if
! CHECK:                 hlfir.assign
! CHECK:                 fir.coordinate_of %[[SARG]], $y
! CHECK:                 fir.store %{{.*}} : !fir.ref<i64>

! -- Copy-back after wsloop (guarded: only store if index >= 0) ----------------
! CHECK:           omp.single {
! CHECK:             fir.coordinate_of %[[STRUCT]], {{[xy]}}
! CHECK:             fir.load
! CHECK:             fir.coordinate_of %[[STRUCT]], ${{[xy]}}
! CHECK:             fir.load
! CHECK:             arith.cmpi sge
! CHECK:             fir.if
! CHECK:               fir.store
! CHECK:             }
! CHECK:             fir.coordinate_of %[[STRUCT]], {{[xy]}}
! CHECK:             fir.load
! CHECK:             fir.coordinate_of %[[STRUCT]], ${{[xy]}}
! CHECK:             fir.load
! CHECK:             arith.cmpi sge
! CHECK:             fir.if
! CHECK:               fir.store
! CHECK:             }
