!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! A user-defined reduction shadows a same-named intrinsic reduction only for
! the types it is declared for. Here "max" is declared for INTEGER, but the
! reduction clause uses a REAL variable, for which no user-defined reduction
! exists. The clause must therefore bind to the implicit intrinsic real max,
! not to the (integer) user-defined reduction.

subroutine test(rr, a)
  real :: rr, a(10)
  integer :: i
!$omp declare reduction(max:integer:omp_out=omp_out+omp_in) initializer(omp_priv=0)
  rr = 0.0
!$omp parallel do reduction(max:rr)
  do i=1,10
     rr = max(rr, a(i))
  end do
end subroutine

! The intrinsic real max reduction is used for the real variable.
! CHECK: omp.declare_reduction @[[MAXF:max_f32]] : f32 init {

! CHECK-LABEL: func.func @_QPtest
! CHECK: omp.wsloop {{.*}}reduction(@[[MAXF]] %{{.*}} -> %{{.*}} : !fir.ref<f32>)
