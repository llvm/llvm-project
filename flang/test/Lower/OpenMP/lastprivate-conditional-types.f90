! Test lowering of `lastprivate(conditional:)` for the supported scalar
! intrinsic type categories other than default integer: real, logical, complex
! (and a non-default integer kind).  The packed reduction struct must carry a
! value field of the variable's own type plus an i64 index field per variable.

! RUN: bbc -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s

subroutine test_conditional_lp_types(n, r, lg, k8, cx)
  implicit none
  integer, intent(in) :: n
  real, intent(inout) :: r
  logical, intent(inout) :: lg
  integer(8), intent(inout) :: k8
  complex, intent(inout) :: cx
  integer :: i

  !$omp parallel do lastprivate(conditional: r, lg, k8, cx)
  do i = 1, n
    if (mod(i, 3) == 0) r = real(i)
    if (mod(i, 4) == 0) lg = (mod(i, 2) == 0)
    if (mod(i, 5) == 0) k8 = int(i, 8)
    if (mod(i, 6) == 0) cx = cmplx(real(i), 1.0)
  end do
  !$omp end parallel do
end subroutine

! Value fields keep each variable's own type; index fields are i64.
! CHECK-LABEL: omp.declare_reduction @lp_cond_byref_rec__lp_cond_t
! CHECK-SAME:    {r:f32,lg:!fir.logical<4>,k8:i64,cx:complex<f32>,$r:i64,$lg:i64,$k8:i64,$cx:i64}
! CHECK:       init {
! CHECK-DAG:     arith.constant -1 : i64
! CHECK:       }
! CHECK:       combiner {
! Four (value,index) pairs -> four sgt comparisons on i64 index fields.
! CHECK-COUNT-4:   arith.cmpi sgt, %{{.*}}, %{{.*}} : i64
! CHECK:       }

! CHECK-LABEL: func.func @_QPtest_conditional_lp_types
! CHECK:         omp.wsloop
! CHECK-SAME:      reduction(byref @lp_cond_byref_rec__lp_cond_t
! Guarded copy-back for each variable in an omp.single sibling.
! CHECK:         omp.single {
! CHECK-COUNT-4:   arith.cmpi sge, %{{.*}}, %{{.*}} : i64
! CHECK:           omp.terminator
! CHECK:         }
