!! The main point of this test is to check that the code compiles at all, so the
!! checking is not very detailed. Not hitting an assert, crashing or otherwise failing
!! to compile is the key point. Also, emitting llvm is required for this to happen.
! RUN: %flang_fc1 -emit-llvm -fopenmp -o - %s 2>&1 | FileCheck %s
subroutine proc
  implicit none
  real(8),allocatable :: F(:)
  real(8),allocatable :: A(:)

!$omp parallel private(A) reduction(+:F)
  allocate(A(10))
!$omp end parallel
end subroutine proc

!CHECK-LABEL: define void @proc_()
!CHECK: call void
!CHECK-SAME: @__kmpc_fork_call(ptr {{.*}}, i32 1, ptr @[[OMP_PAR:.*]], {{.*}})

!CHECK: define internal void @[[OMP_PAR]]
!CHECK: omp.par.region8:
!CHECK-NEXT: call ptr @malloc
!CHECK-SAME: i64 10

