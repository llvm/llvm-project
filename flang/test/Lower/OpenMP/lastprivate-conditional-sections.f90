! Test lowering of `lastprivate(conditional:)` on an omp sections construct
! with multiple variables.  The lowering must:
!   1. Build a packed struct type {val, val, ..., idx, idx, ...}
!   2. Create an omp.declare_reduction with identity 0 / -1
!   3. Inject the struct as a by-ref reduction variable on the sections
!   4. Rewrite assignments to use struct value fields + store constant section
!      index (0, 1, ...) into the index fields
!   5. Copy back the winning values after the sections

! RUN: bbc -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s

subroutine test_conditional_lp_sections(x, y)
  implicit none
  integer, intent(inout) :: x, y

  !$omp parallel sections lastprivate(conditional: x, y)
    !$omp section
    x = 10
    y = 20

    !$omp section
    x = 30
    y = 40
  !$omp end parallel sections
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

! -- Struct alloca + init before parallel -------------------------------------
! CHECK-LABEL: func.func @_QPtest_conditional_lp_sections
! CHECK:         %[[STRUCT:.*]] = fir.alloca !fir.type<_lp_cond_t.{{l[0-9]+\.[0-9]+}}{x:i32,y:i32,$x:i64,$y:i64}>
! CHECK-DAG:     arith.constant 0 : i32
! CHECK-DAG:     arith.constant -1 : i64

! -- Sections carries the struct as a by-ref reduction ------------------------
! CHECK:         omp.parallel {
! CHECK:           omp.sections
! CHECK-SAME:        reduction(byref @lp_cond_byref_rec__lp_cond_t
! CHECK-SAME:          %[[STRUCT]]

! -- Section 0: index constant hoisted to entry, stored after each assign -----
! CHECK:             omp.section {
! CHECK:               %[[IDX0:.*]] = arith.constant 0 : i64
! CHECK:               hlfir.assign
! CHECK:               fir.store %[[IDX0]]
! CHECK:               hlfir.assign
! CHECK:               fir.store %[[IDX0]]

! -- Section 1: index constant hoisted to entry, stored after each assign -----
! CHECK:             omp.section {
! CHECK:               %[[IDX1:.*]] = arith.constant 1 : i64
! CHECK:               hlfir.assign
! CHECK:               fir.store %[[IDX1]]
! CHECK:               hlfir.assign
! CHECK:               fir.store %[[IDX1]]

! -- Copy-back after sections (guarded: only store if index >= 0) --------------
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
