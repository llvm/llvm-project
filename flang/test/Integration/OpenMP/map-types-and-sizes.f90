!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!RUN: %flang_fc1 -emit-llvm -fopenmp -flang-deprecated-no-hlfir %s -o - | FileCheck %s

!===============================================================================
! Check MapTypes for target implicit captures
!===============================================================================

!CHECK: @.offload_sizes = private unnamed_addr constant [1 x i64] [i64 4]
!CHECK: @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 800]
subroutine mapType_scalar
  integer :: a
  !$omp target
     a = 10
  !$omp end target
end subroutine mapType_scalar

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [1 x i64] [i64 4096]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [1 x i64] [i64 547]
subroutine mapType_array
  integer :: a(1024)
  !$omp target
     a(10) = 20
  !$omp end target
end subroutine mapType_array

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [3 x i64] [i64 0, i64 24, i64 4]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [3 x i64] [i64 32, i64 281474976710657, i64 281474976711187]
subroutine mapType_ptr
  integer, pointer :: a
  !$omp target
     a = 10
  !$omp end target
end subroutine mapType_ptr

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [3 x i64] [i64 0, i64 24, i64 4]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [3 x i64] [i64 32, i64 281474976710657, i64 281474976711187]
subroutine mapType_allocatable
  integer, allocatable :: a
  allocate(a)
  !$omp target
     a = 10
  !$omp end target
  deallocate(a)
end subroutine mapType_allocatable

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [3 x i64] [i64 0, i64 24, i64 4]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [3 x i64] [i64 32, i64 281474976710657, i64 281474976710675]
subroutine mapType_ptr_explicit
  integer, pointer :: a
  !$omp target map(tofrom: a)
     a = 10
  !$omp end target
end subroutine mapType_ptr_explicit

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [3 x i64] [i64 0, i64 24, i64 4]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [3 x i64] [i64 32, i64 281474976710657, i64 281474976710675]
subroutine mapType_allocatable_explicit
  integer, allocatable :: a
  allocate(a)
  !$omp target map(tofrom: a)
     a = 10
  !$omp end target
  deallocate(a)
end subroutine mapType_allocatable_explicit

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [2 x i64] [i64 8, i64 4]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [2 x i64] [i64 544, i64 800]
subroutine mapType_c_ptr
  use iso_c_binding, only : c_ptr, c_loc
  type(c_ptr) :: a
  integer, target :: b
  !$omp target
     a = c_loc(b)
  !$omp end target
end subroutine mapType_c_ptr

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [1 x i64] [i64 1]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [1 x i64] [i64 800]
subroutine mapType_char
  character :: a
  !$omp target
     a = 'b'
  !$omp end target
end subroutine mapType_char

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_ptr_explicit_{{.*}}
!CHECK: %[[ALLOCA:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8 }, i64 1, align 8
!CHECK: %[[ALLOCA_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8 }, ptr %[[ALLOCA]], i32 1
!CHECK: %[[ALLOCA_GEP_INT:.*]] = ptrtoint ptr %[[ALLOCA_GEP]] to i64
!CHECK: %[[ALLOCA_INT:.*]] = ptrtoint ptr %[[ALLOCA]] to i64
!CHECK: %[[SIZE_DIFF:.*]] = sub i64 %[[ALLOCA_GEP_INT]], %[[ALLOCA_INT]]
!CHECK: %[[DIV:.*]] = sdiv exact i64 %[[SIZE_DIFF]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
!CHECK: %[[OFFLOAD_SIZE_ARR:.*]] = getelementptr inbounds [3 x i64], ptr %.offload_sizes, i32 0, i32 0
!CHECK: store i64 %[[DIV]], ptr %[[OFFLOAD_SIZE_ARR]], align 8


!CHECK-LABEL: define {{.*}} @{{.*}}maptype_allocatable_explicit_{{.*}}
!CHECK: %[[ALLOCA:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8 }, i64 1, align 8
!CHECK: %[[ALLOCA_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8 }, ptr %[[ALLOCA]], i32 1
!CHECK: %[[ALLOCA_GEP_INT:.*]] = ptrtoint ptr %[[ALLOCA_GEP]] to i64
!CHECK: %[[ALLOCA_INT:.*]] = ptrtoint ptr %[[ALLOCA]] to i64
!CHECK: %[[SIZE_DIFF:.*]] = sub i64 %[[ALLOCA_GEP_INT]], %[[ALLOCA_INT]]
!CHECK: %[[DIV:.*]] = sdiv exact i64 %[[SIZE_DIFF]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
!CHECK: %[[OFFLOAD_SIZE_ARR:.*]] = getelementptr inbounds [3 x i64], ptr %.offload_sizes, i32 0, i32 0
!CHECK: store i64 %[[DIV]], ptr %[[OFFLOAD_SIZE_ARR]], align 8
