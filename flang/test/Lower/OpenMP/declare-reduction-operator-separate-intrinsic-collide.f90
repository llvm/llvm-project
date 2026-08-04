! Cross-module intrinsic-operator reduction collision: two modules m1 and m2 each
! declare reduction(+:t) for the SAME derived type t (imported from a common base
! module) with DISTINCT combiners (+ vs -). A consumer uses m1 in one subroutine
! and m2 in another and reduces with the intrinsic operator. Each imported
! reduction must be materialized under its own module-scoped name and bound, so
! the two subroutines lower to two DISTINCT ops keyed by their source module,
! not one shared op that would run one variable through the other's combiner.
! Before intrinsic-operator reductions were module-scoped they collided on the
! one global builtin name (@add_reduction_...).
! https://github.com/llvm/llvm-project/issues/207255

! RUN: rm -rf %t && split-file %s %t && cd %t
! RUN: %flang_fc1 -fsyntax-only -fopenmp base.f90
! RUN: %flang_fc1 -fsyntax-only -fopenmp m1.f90
! RUN: %flang_fc1 -fsyntax-only -fopenmp m2.f90
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 use.f90 -o - | FileCheck use.f90

!--- base.f90
module base
  type :: t
    integer :: v = 0
  end type
end module

!--- m1.f90
module m1
  use base, only: t
  !$omp declare reduction(+:t:omp_out%v=omp_out%v+omp_in%v) initializer(omp_priv=t(0))
end module

!--- m2.f90
module m2
  use base, only: t
  !$omp declare reduction(+:t:omp_out%v=omp_out%v-omp_in%v) initializer(omp_priv=t(0))
end module

!--- use.f90
! Two distinct scoped ops keyed by their source module (m1 vs m2); the global
! builtin name is never used for these user reductions, and no clause aborts.
! CHECK-DAG: omp.declare_reduction @"{{_QQ[A-Za-z0-9_.]*m1[A-Za-z0-9_.]*op\.\+[A-Za-z0-9_.]*}}" : !fir.ref
! CHECK-DAG: omp.declare_reduction @"{{_QQ[A-Za-z0-9_.]*m2[A-Za-z0-9_.]*op\.\+[A-Za-z0-9_.]*}}" : !fir.ref
! CHECK-NOT: @add_reduction
! CHECK-NOT: not yet implemented

! Each subroutine binds the op scoped to the module it imported.
! CHECK-LABEL: func.func @_QPs1
! CHECK: omp.wsloop {{.*}}reduction(byref @"{{_QQ[A-Za-z0-9_.]*m1[A-Za-z0-9_.]*op\.\+[A-Za-z0-9_.]*}}"
! CHECK-LABEL: func.func @_QPs2
! CHECK: omp.wsloop {{.*}}reduction(byref @"{{_QQ[A-Za-z0-9_.]*m2[A-Za-z0-9_.]*op\.\+[A-Za-z0-9_.]*}}"
subroutine s1(x)
  use base, only: t
  use m1
  type(t) :: x
  integer :: i
  !$omp parallel do reduction(+:x)
  do i = 1, 10
    x%v = x%v + 1
  end do
end subroutine

subroutine s2(x)
  use base, only: t
  use m2
  type(t) :: x
  integer :: i
  !$omp parallel do reduction(+:x)
  do i = 1, 10
    x%v = x%v + 1
  end do
end subroutine
