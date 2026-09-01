! Two user intrinsic-operator declare reductions for the SAME operator (+) and
! SAME derived type, declared in two different scopes (two module subroutines)
! with DISTINCT combiners, must lower to two DISTINCT omp.declare_reduction ops,
! each bound by its own scope's reduction clause. Before user intrinsic-operator
! reductions were module-scoped they were named by the global builtin name
! (@add_reduction_...), so the two declarations collided on one op and one
! subroutine silently ran the other's combiner. The intrinsic-operator analog of
! the named-reduction test for issue #181270.
! https://github.com/llvm/llvm-project/issues/207255

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

module m
  type :: t
    integer :: x = 0
  end type
contains
  subroutine sum_scope
    !$omp declare reduction(+:t:omp_out%x=omp_out%x+omp_in%x) initializer(omp_priv=t(0))
    type(t) :: a
    integer :: i
    a = t(0)
    !$omp parallel do reduction(+:a)
    do i = 1, 10
      a%x = a%x + 1
    end do
    !$omp end parallel do
    print *, a%x
  end subroutine

  subroutine prod_scope
    !$omp declare reduction(+:t:omp_out%x=omp_out%x*omp_in%x) initializer(omp_priv=t(1))
    type(t) :: b
    integer :: i
    b = t(1)
    !$omp parallel do reduction(+:b)
    do i = 1, 10
      b%x = b%x * 2
    end do
    !$omp end parallel do
    print *, b%x
  end subroutine
end module m

! Two distinct scoped ops, one per subroutine scope. The global builtin name is
! never used for a user reduction (the collision-fix signal).
! CHECK-DAG: omp.declare_reduction @"{{_QQ[A-Za-z0-9_.]*sum_scope[A-Za-z0-9_.]*op\.\+[A-Za-z0-9_.]*}}" : !fir.ref
! CHECK-DAG: omp.declare_reduction @"{{_QQ[A-Za-z0-9_.]*prod_scope[A-Za-z0-9_.]*op\.\+[A-Za-z0-9_.]*}}" : !fir.ref
! CHECK-NOT: @add_reduction

! Each subroutine's clause binds the op scoped to its own subroutine, so the two
! loops run distinct combiners.
! CHECK-LABEL: func.func @_QMmPsum_scope
! CHECK: omp.wsloop {{.*}}reduction(byref @"[[SUM:_QQ[A-Za-z0-9_.]*sum_scope[A-Za-z0-9_.]*op\.\+[A-Za-z0-9_.]*]]"
! CHECK-LABEL: func.func @_QMmPprod_scope
! CHECK: omp.wsloop {{.*}}reduction(byref @"[[PROD:_QQ[A-Za-z0-9_.]*prod_scope[A-Za-z0-9_.]*op\.\+[A-Za-z0-9_.]*]]"
