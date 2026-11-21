! Test lowering of a list item that is both `firstprivate` and
! `lastprivate(conditional:)` on a worksharing construct.
!
! Under the worksharing private-copy lowering, the item gets an ordinary
! (firstprivate) private copy -- its in-loop working value, initialized from the
! original -- while a separate reduction struct is the conditional-last
! accumulator.  A guarded commit copies the working value into the accumulator
! when the current canonical index is highest so far; a guarded copy-back writes
! the accumulator to the original after the loop.

! RUN: bbc -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s

subroutine test_fp_conditional_lp(n, x)
  implicit none
  integer, intent(in) :: n
  integer :: x
  integer :: k

  !$omp parallel do firstprivate(x) lastprivate(conditional: x)
  do k = 1, n
    if (mod(k, 2) == 0) x = x + k
  end do
  !$omp end parallel do
end subroutine

! CHECK-LABEL: func.func @_QPtest_fp_conditional_lp
! x is an ordinary firstprivate copy (working value); the struct is the
! conditional-last accumulator carried as a by-ref reduction.
! CHECK:         omp.wsloop private(@{{.*}}Ex_firstprivate
! CHECK-SAME:      reduction(byref @lp_cond_byref_rec__lp_cond_t{{.*}} -> %[[SARG:.*]] :
! -- Loop body: guarded commit into the accumulator. -------------------------
! CHECK:           fir.if
! CHECK:             hlfir.assign
! CHECK:             fir.coordinate_of %[[SARG]], x
! CHECK:             fir.coordinate_of %[[SARG]], $x
! CHECK:             arith.cmpi sge, %{{.*}}, %{{.*}} : i64
! CHECK:             fir.if
! CHECK:               fir.store %{{.*}} : !fir.ref<i32>
! CHECK:               fir.store %{{.*}} : !fir.ref<i64>
! -- Guarded copy-back after the loop. ---------------------------------------
! CHECK:         omp.single {
! CHECK:           arith.cmpi sge, %{{.*}}, %{{.*}} : i64
! CHECK:           fir.if
! CHECK:             fir.store
! CHECK:           omp.terminator

! =====================================================================
! Mixed: one item is firstprivate + conditional (a), another is conditional
! only (b).  Both get a private copy; the struct accumulates both.
! =====================================================================
subroutine test_fp_cond_mixed(n, a, b)
  implicit none
  integer, intent(in) :: n
  integer :: a, b
  integer :: k

  !$omp parallel do firstprivate(a) lastprivate(conditional: a, b)
  do k = 1, n
    if (mod(k, 2) == 0) a = a + k
    if (mod(k, 3) == 0) b = k
  end do
  !$omp end parallel do
end subroutine

! CHECK-LABEL: func.func @_QPtest_fp_cond_mixed
! a is firstprivate; b is an ordinary private copy; both feed the reduction.
! CHECK:         omp.wsloop private(@{{.*}}Ea_firstprivate
! CHECK-SAME:      reduction(byref @lp_cond_byref_rec__lp_cond_t

! =====================================================================
! Coexistence: a firstprivate item (q) and a distinct conditional item (p) on
! the same construct.  q keeps its firstprivate privatizer; p uses the
! conditional-lastprivate reduction struct (and its own private copy).
! =====================================================================
subroutine test_fp_cond_coexist(n, p, q)
  implicit none
  integer, intent(in) :: n
  integer :: p, q
  integer :: k

  !$omp parallel do firstprivate(q) lastprivate(conditional: p)
  do k = 1, n
    if (mod(k, 2) == 0) p = q + k
  end do
  !$omp end parallel do
end subroutine

! CHECK-LABEL: func.func @_QPtest_fp_cond_coexist
! CHECK:         omp.wsloop private(@{{.*}}Eq_firstprivate
! CHECK-SAME:      reduction(byref @lp_cond_byref_rec__lp_cond_t
