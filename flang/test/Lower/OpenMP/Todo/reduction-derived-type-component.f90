! Has to be v5.2 because in some earlier standards this is a semantic error
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s
subroutine intrinsic_reduction
  type local
     integer alpha
  end type local
  type(local) a
! CHECK: not yet implemented: Reduction symbol has no definition
!$omp parallel reduction(+:a%alpha)
!$omp end parallel
end subroutine intrinsic_reduction
