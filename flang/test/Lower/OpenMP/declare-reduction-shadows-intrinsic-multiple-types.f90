! A user-defined declare reduction whose identifier shadows an intrinsic
! reduction name (max/min/iand/ior/ieor) and lists several types must lower to
! one omp.declare_reduction op per listed type, and each reduction clause must
! bind the op for its variable's type (rather than falling back to the built-in
! intrinsic reduction). This is the intrinsic-shadowing counterpart of the
! operator and named multi-type paths. Single-type shadowing is handled by
! declare-reduction-shadows-intrinsic.f90; here the declaration lists both
! integer and real. See https://github.com/llvm/llvm-project/issues/207255.

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

program p
  integer :: i, isum
  real :: rsum

  !$omp declare reduction(max: integer, real : omp_out = max(omp_out, omp_in)) &
  !$omp   initializer(omp_priv = omp_orig)

  isum = 0
  rsum = 0.0

  !$omp parallel do reduction(max: isum)
  do i = 1, 5
    isum = max(isum, i)
  end do

  !$omp parallel do reduction(max: rsum)
  do i = 1, 5
    rsum = max(rsum, real(i))
  end do

  print *, isum, rsum
end program

! The shadowing user reduction is named "op.max" (MangleSpecialFunctions), and
! being multi-type it gets one type-suffixed op per listed type. There must be
! no single type-less op that both types fold onto.
! CHECK-NOT: omp.declare_reduction @_QQFop.max :
! CHECK-DAG: omp.declare_reduction @[[MAXI:_QQFop.max_i32]] : i32
! CHECK-DAG: omp.declare_reduction @[[MAXR:_QQFop.max_f32]] : f32

! Each loop binds the op for its own variable's type, by full name (not the
! shared @_QQFop.max_ prefix, and not the built-in max reduction).
! CHECK-DAG: reduction(@[[MAXI]] {{[^)]*}}!fir.ref<i32>
! CHECK-DAG: reduction(@[[MAXR]] {{[^)]*}}!fir.ref<f32>
