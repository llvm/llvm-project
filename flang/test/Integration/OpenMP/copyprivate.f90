!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!RUN: %flang_fc1 -emit-llvm -fopenmp %s -o - | FileCheck %s

!CHECK-DAG: define void @_copy_box_Uxi32(ptr %{{.*}}, ptr %{{.*}})
!CHECK-DAG: define void @_copy_10xi32(ptr %{{.*}}, ptr %{{.*}})
!CHECK-DAG: define void @_copy_i64(ptr %{{.*}}, ptr %{{.*}})
!CHECK-DAG: define void @_copy_box_Uxi64(ptr %{{.*}}, ptr %{{.*}})
!CHECK-DAG: define void @_copy_f32(ptr %{{.*}}, ptr %{{.*}})
!CHECK-DAG: define void @_copy_2x3xf32(ptr %{{.*}}, ptr %{{.*}})
!CHECK-DAG: define void @_copy_z32(ptr %{{.*}}, ptr %{{.*}})
!CHECK-DAG: define void @_copy_10xz32(ptr %{{.*}}, ptr %{{.*}})
!CHECK-DAG: define void @_copy_l32(ptr %{{.*}}, ptr %{{.*}})
!CHECK-DAG: define void @_copy_5xl32(ptr %{{.*}}, ptr %{{.*}})
!CHECK-DAG: define void @_copy_c8x8(ptr %{{.*}}, ptr %{{.*}})
!CHECK-DAG: define void @_copy_10xc8x8(ptr %{{.*}}, ptr %{{.*}})
!CHECK-DAG: define void @_copy_c16x5(ptr %{{.*}}, ptr %{{.*}})
!CHECK-DAG: define void @_copy_rec__QFtest_typesTdt(ptr %{{.*}}, ptr %{{.*}})
!CHECK-DAG: define void @_copy_box_heap_Uxi32(ptr %{{.*}}, ptr %{{.*}})
!CHECK-DAG: define void @_copy_box_ptr_Uxc8x9(ptr %{{.*}}, ptr %{{.*}})

!CHECK-LABEL: define void @_copy_i32(
!CHECK-SAME:                         ptr %[[DST:.*]], ptr %[[SRC:.*]]){{.*}} {
!CHECK-NEXT:    %[[SRC_VAL:.*]] = load i32, ptr %[[SRC]]
!CHECK-NEXT:    store i32 %[[SRC_VAL]], ptr %[[DST]]
!CHECK-NEXT:    ret void
!CHECK-NEXT:  }

!CHECK-LABEL: define internal void @test_scalar_..omp_par({{.*}})
!CHECK:         %[[J:.*]] = alloca i32, i64 1
!CHECK:         %[[I:.*]] = alloca i32, i64 1
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

subroutine test_types(a, n)
  integer :: a(:), n
  integer(4) :: i4, i4a(10)
  integer(8) :: i8, i8a(n)
  real :: r, ra(2, 3)
  complex :: z, za(10)
  logical :: l, la(5)
  character(kind=1, len=8) :: c1, c1a(10)
  character(kind=2, len=5) :: c2

  type dt
    integer :: i
    real :: r
  end type
  type(dt) :: t

  integer, allocatable :: aloc(:)
  character(kind=1, len=9), pointer :: ptr(:)

  !$omp parallel private(a, i4, i4a, i8, i8a, r, ra, z, za, l, la, c1, c1a, c2, t, aloc, ptr)
  !$omp single
  !$omp end single copyprivate(a, i4, i4a, i8, i8a, r, ra, z, za, l, la, c1, c1a, c2, t, aloc, ptr)
  !$omp end parallel
end subroutine
