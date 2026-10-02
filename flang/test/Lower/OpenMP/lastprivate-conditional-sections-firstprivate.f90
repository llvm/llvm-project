! Test lowering of a list item that is both `firstprivate` and
! `lastprivate(conditional:)` on a SECTIONS construct (the sections seed-store
! path, analogous to the worksharing-do path in
! lastprivate-conditional-firstprivate.f90).

! RUN: bbc -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s

subroutine test_fp_cond_sections(x)
  implicit none
  integer :: x

  !$omp parallel sections firstprivate(x) lastprivate(conditional: x)
  !$omp section
    x = x + 5
  !$omp end parallel sections
end subroutine

! The struct is the sole binding for x: no ordinary firstprivate privatizer.
! CHECK-NOT: omp.private {{.*}}firstprivate

! CHECK-LABEL: omp.declare_reduction @lp_cond_byref_rec__lp_cond_t
! Init region seeds the value field from the seed struct (ompOrig) per thread.
! CHECK:       init {
! CHECK:         ^bb0(%[[ORIG:.*]]: {{.*}}, %[[PRIV:.*]]: {{.*}}):
! CHECK:         fir.coordinate_of %[[PRIV]], x
! CHECK:         fir.coordinate_of %[[ORIG]], x
! CHECK:       }

! CHECK-LABEL: func.func @_QPtest_fp_cond_sections
! Seed store before the region: struct value field = original x.
! CHECK:         %[[XD:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QF{{.*}}Ex"}
! CHECK:         fir.coordinate_of %[[S:.*]], $x
! CHECK:         %[[MX:.*]] = fir.coordinate_of %[[S]], x
! CHECK:         %[[SEED:.*]] = fir.load %[[XD]]#0
! CHECK:         fir.store %[[SEED]] to %[[MX]]
! The conditional reduction is carried on omp.sections; guarded copy-back after.
! CHECK:         omp.sections
! CHECK-SAME:      reduction(byref @lp_cond_byref_rec__lp_cond_t
! CHECK:           omp.single {
! CHECK:             arith.cmpi sge, %{{.*}}, %{{.*}} : i64
! CHECK:             omp.terminator
! CHECK:           }
