!! Make sure that mixture of by-ref and by-val reductions work all the way
!! to LLVM-IR code.

! RUN: %flang_fc1 -emit-llvm -fopenmp -o - %s 2>&1 \
! RUN: | FileCheck %s

subroutine proc
  implicit none
  real(8),allocatable :: F(:)
  real(8),allocatable :: A(:)
   
  integer :: I

!$omp parallel private(A) reduction(+:F,I)
  allocate(A(10))
!$omp end parallel
end subroutine proc

!CHECK-LABEL: define void @proc_()
!CHECK: call void (ptr, i32, ptr, ...)
!CHECK-SAME: @__kmpc_fork_call(ptr {{.*}}, i32 1, ptr @[[OMP_PAR:.*]], {{.*}})

!CHECK: define internal void @[[OMP_PAR]](ptr {{.*}} %[[TID_ADDR:.*]], ptr noalias 
!CHECK:  %[[TID_LOCAL:.*]] = alloca i32
!CHECK:  %[[TID:.*]] = load i32, ptr %[[TID_ADDR]]
!CHECK:  store i32 %[[TID]], ptr %[[TID_LOCAL]]
!CHECK:  %[[F_priv:.*]] = alloca ptr
!CHECK:  %[[I_priv:.*]] = alloca i32

!CHECK: omp.par.region:

!CHECK: omp.reduction.init:
!CHECK:  store ptr %{{.*}}, ptr %[[F_priv]]
!CHECK:  store i32 0, ptr %[[I_priv]]
!CHECK:  br label %[[MALLOC_BB:.*]]

!CHECK: [[MALLOC_BB]]:
!CHECK-NOT: omp.par.{{.*}}:
!CHECK: call ptr @malloc
!CHECK-SAME: i64 10

!CHECK: %[[RED_ARR_0:.*]] = getelementptr inbounds [2 x ptr], ptr %red.array, i64 0, i64 0
!CHECK: store ptr %[[F_priv]], ptr %[[RED_ARR_0:.*]]
!CHECK: %[[RED_ARR_1:.*]] = getelementptr inbounds [2 x ptr], ptr %red.array, i64 0, i64 1
!CHECK: store ptr %[[I_priv]], ptr %[[RED_ARR_1]]

!CHECK: omp.par.pre_finalize:                             ; preds = %reduce.finalize
!CHECK:  %{{.*}} = load ptr, ptr %[[F_priv]]
!CHECK:  br label %omp.reduction.cleanup

!CHECK: omp.reduction.cleanup:
!CHECK:  br i1 %{{.*}}, label %[[OMP_FREE:.*]], label %{{.*}}

!CHECK: [[OMP_FREE]]:
!CHECK: call void @free
