!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!RUN: %flang_fc1 -emit-llvm -fopenmp %s -o - | FileCheck %s

!CHECK-LABEL: define void @_copy_i32(
!CHECK-SAME:                         ptr %[[DST:.*]], ptr %[[SRC:.*]]) {
!CHECK-NEXT:    %[[SRC_VAL:.*]] = load i32, ptr %[[SRC]]
!CHECK-NEXT:    store i32 %[[SRC_VAL]], ptr %[[DST]]
!CHECK-NEXT:    ret void
!CHECK-NEXT:  }

!CHECK-LABEL: define internal void @test_scalar_..omp_par({{.*}})
!CHECK:         %[[I:.*]] = alloca i32, i64 1
!CHECK:         %[[J:.*]] = alloca i32, i64 1
!CHECK:         %[[DID_IT:.*]] = alloca i32
!CHECK:         store i32 0, ptr %[[DID_IT]]
!CHECK:         %[[THREAD_NUM1:.*]] = call i32 @__kmpc_global_thread_num(ptr @[[LOC:.*]])
!CHECK:         %[[RET:.*]] = call i32 @__kmpc_single({{.*}})
!CHECK:         %[[NOT_ZERO:.*]] = icmp ne i32 %[[RET]], 0
!CHECK:         br i1 %[[NOT_ZERO]], label %[[OMP_REGION_BODY:.*]], label %[[OMP_REGION_END:.*]]

!CHECK:       [[OMP_REGION_END]]:
!CHECK:         %[[THREAD_NUM2:.*]] = call i32 @__kmpc_global_thread_num(ptr @[[LOC:.*]])
!CHECK:         %[[DID_IT_VAL:.*]] = load i32, ptr %[[DID_IT]]
!CHECK:         call void @__kmpc_copyprivate(ptr @[[LOC]], i32 %[[THREAD_NUM2]], i64 0, ptr %[[I]], ptr @_copy_i32, i32 %[[DID_IT_VAL]])
!CHECK:         %[[THREAD_NUM3:.*]] = call i32 @__kmpc_global_thread_num(ptr @[[LOC]])
!CHECK:         %[[DID_IT_VAL2:.*]] = load i32, ptr %[[DID_IT]]
!CHECK:         call void @__kmpc_copyprivate(ptr @[[LOC]], i32 %[[THREAD_NUM3]], i64 0, ptr %[[J]], ptr @_copy_i32, i32 %[[DID_IT_VAL2]])

!CHECK:       [[OMP_REGION_BODY]]:
!CHECK:         br label %[[OMP_SINGLE_REGION:.*]]
!CHECK:       [[OMP_SINGLE_REGION]]:
!CHECK:         store i32 11, ptr %[[I]]
!CHECK:         store i32 22, ptr %[[J]]
!CHECK:         br label %[[OMP_REGION_CONT3:.*]]
!CHECK:       [[OMP_REGION_CONT3:.*]]:
!CHECK:         store i32 1, ptr %[[DID_IT]]
!CHECK:         call void @__kmpc_end_single(ptr @[[LOC]], i32 %[[THREAD_NUM1]])
!CHECK:         br label %[[OMP_REGION_END]]
subroutine test_scalar()
  integer :: i, j

  !$omp parallel private(i, j)
  !$omp single
  i = 11
  j = 22
  !$omp end single copyprivate(i, j)
  !$omp end parallel
end subroutine
