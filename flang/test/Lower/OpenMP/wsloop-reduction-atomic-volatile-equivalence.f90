! A VOLATILE EQUIVALENCE member lowers to a fir.ptr without the volatile
! qualifier. The reduction must still be treated as volatile: named
! @add_reduction_volatile_i32, with no atomic region.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --strict-fir-volatile-verifier -o - %s | FileCheck %s --implicit-check-not=llvm.atomicrmw

! CHECK:       omp.declare_reduction @add_reduction_volatile_i32 : i32
! CHECK-LABEL: func.func @_QPveq
! CHECK:         omp.wsloop {{.*}}reduction(@add_reduction_volatile_i32 %{{.*}} -> %{{.*}} : !fir.ref<i32, volatile>)

subroutine veq(n)
  integer :: n, i
  integer, volatile :: vi
  real :: x
  equivalence (vi, x)
  vi = 0
  !$omp parallel do reduction(+:vi)
  do i = 1, n
    vi = vi + 1
  end do
end subroutine

! An ASSOCIATE name for a component of a VOLATILE EQUIVALENCE object: the
! volatility comes from the selector's base object.
! CHECK-LABEL: func.func @_QPeq_comp
! CHECK:         omp.wsloop {{.*}}reduction(@add_reduction_volatile_i32 %{{.*}} -> %{{.*}} : !fir.ref<i32, volatile>)

subroutine eq_comp(n)
  type :: t
    sequence
    integer :: c
  end type
  integer :: n, i, x
  type(t), volatile :: v
  equivalence (v, x)
  v%c = 0
  associate (a => v%c)
    !$omp parallel do reduction(+:a)
    do i = 1, n
      a = a + 1
    end do
  end associate
end subroutine
