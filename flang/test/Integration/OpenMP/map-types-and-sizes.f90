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
 
!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [1 x i64] [i64 48]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [1 x i64] [i64 547]
subroutine mapType_derived_implicit
  type :: scalar_and_array
    real(4) :: real
    integer(4) :: array(10)
    integer(4) :: int
  end type scalar_and_array
  type(scalar_and_array) :: scalar_arr 
  
  !$omp target
     scalar_arr%int = 1
  !$omp end target
end subroutine mapType_derived_implicit

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [1 x i64] [i64 48]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [1 x i64] [i64 35]
subroutine mapType_derived_explicit
  type :: scalar_and_array
    real(4) :: real
    integer(4) :: array(10)
    integer(4) :: int
  end type scalar_and_array
  type(scalar_and_array) :: scalar_arr 
  
  !$omp target map(tofrom: scalar_arr)
     scalar_arr%int = 1
  !$omp end target
end subroutine mapType_derived_explicit

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [1 x i64] [i64 40]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [1 x i64] [i64 35]
subroutine mapType_derived_explicit_single_member
  type :: scalar_and_array
    real(4) :: real
    integer(4) :: array(10)
    integer(4) :: int
  end type scalar_and_array
  type(scalar_and_array) :: scalar_arr 
  
  !$omp target map(tofrom: scalar_arr%array)
     scalar_arr%array(1) = 1
  !$omp end target
end subroutine mapType_derived_explicit_single_member

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [3 x i64] [i64 0, i64 4, i64 4]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [3 x i64] [i64 32, i64 281474976710659, i64 281474976710659]
subroutine mapType_derived_explicit_multiple_members
  type :: scalar_and_array
    real(4) :: real
    integer(4) :: array(10)
    integer(4) :: int
  end type scalar_and_array
  type(scalar_and_array) :: scalar_arr 
  
  !$omp target map(tofrom: scalar_arr%int, scalar_arr%real)
     scalar_arr%int = 1
  !$omp end target
end subroutine mapType_derived_explicit_multiple_members

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [1 x i64] [i64 16]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [1 x i64] [i64 35]
subroutine mapType_derived_explicit_member_with_bounds
  type :: scalar_and_array
    real(4) :: real
    integer(4) :: array(10)
    integer(4) :: int
  end type scalar_and_array
  type(scalar_and_array) :: scalar_arr 
  
  !$omp target map(tofrom: scalar_arr%array(2:5))
     scalar_arr%array(3) = 3
  !$omp end target
end subroutine mapType_derived_explicit_member_with_bounds

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [1 x i64] [i64 4]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [1 x i64] [i64 35]
subroutine mapType_derived_explicit_nested_single_member
  type :: nested
    integer(4) :: int
    real(4) :: real
    integer(4) :: array(10)
  end type nested

  type :: scalar_and_array
    real(4) :: real
    integer(4) :: array(10)
    type(nested) :: nest
    integer(4) :: int
  end type scalar_and_array
  type(scalar_and_array) :: scalar_arr 
  
  !$omp target map(tofrom: scalar_arr%nest%real)
    scalar_arr%nest%real = 1
  !$omp end target
end subroutine mapType_derived_explicit_nested_single_member

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [3 x i64] [i64 0, i64 4, i64 4]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [3 x i64] [i64 32, i64 281474976710659, i64 281474976710659]
subroutine mapType_derived_explicit_multiple_nested_members
  type :: nested
    integer(4) :: int
    real(4) :: real
    integer(4) :: array(10)
  end type nested

  type :: scalar_and_array
    real(4) :: real
    integer(4) :: array(10)
    type(nested) :: nest
    integer(4) :: int
  end type scalar_and_array
  type(scalar_and_array) :: scalar_arr 
  
!$omp target map(tofrom: scalar_arr%nest%int, scalar_arr%nest%real)
  scalar_arr%nest%int = 1
!$omp end target
end subroutine mapType_derived_explicit_multiple_nested_members

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [1 x i64] [i64 16]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [1 x i64] [i64 35]
subroutine mapType_derived_explicit_nested_member_with_bounds
  type :: nested
    integer(4) :: int
    real(4) :: real
    integer(4) :: array(10)
  end type nested

  type :: scalar_and_array
    real(4) :: real
    integer(4) :: array(10)
    type(nested) :: nest
    integer(4) :: int
  end type scalar_and_array
  type(scalar_and_array) :: scalar_arr 
  
!$omp target map(tofrom: scalar_arr%nest%array(2:5))
    scalar_arr%nest%array(3) = 3
!$omp end target
end subroutine mapType_derived_explicit_nested_member_with_bounds

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

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [1 x i64] [i64 8]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [1 x i64] [i64 35]
subroutine mapType_common_block
  implicit none
  common /var_common/ var1, var2
  integer :: var1, var2
!$omp target map(tofrom: /var_common/)
  var1 = var1 + 20
  var2 = var2 + 30
!$omp end target
end subroutine mapType_common_block

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [2 x i64] [i64 4, i64 4]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [2 x i64] [i64 35, i64 35]
subroutine mapType_common_block_members
  implicit none
  common /var_common/ var1, var2
  integer :: var1, var2

!$omp target map(tofrom: var1, var2)
  var2 = var1
!$omp end target
end subroutine mapType_common_block_members


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

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_derived_implicit_{{.*}}
!CHECK: %[[ALLOCA:.*]] = alloca %_QFmaptype_derived_implicitTscalar_and_array, i64 1, align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [1 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [1 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr %[[ALLOCA]], ptr %[[OFFLOAD_PTR_ARR]], align 8

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_derived_explicit_{{.*}}
!CHECK: %[[ALLOCA:.*]] = alloca %_QFmaptype_derived_explicitTscalar_and_array, i64 1, align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [1 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [1 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr %[[ALLOCA]], ptr %[[OFFLOAD_PTR_ARR]], align 8

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_derived_explicit_single_member_{{.*}}
!CHECK: %[[ALLOCA:.*]] = alloca %_QFmaptype_derived_explicit_single_memberTscalar_and_array, i64 1, align 8
!CHECK: %[[MEMBER_ACCESS:.*]] = getelementptr %_QFmaptype_derived_explicit_single_memberTscalar_and_array, ptr %[[ALLOCA]], i32 0, i32 1
!CHECK: %[[ARR_OFF:.*]] = getelementptr inbounds [10 x i32], ptr %[[MEMBER_ACCESS]], i64 0, i64 0
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [1 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [1 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr %[[ARR_OFF]], ptr %[[OFFLOAD_PTR_ARR]], align 8

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_derived_explicit_multiple_members_{{.*}}
!CHECK: %[[ALLOCA:.*]] = alloca %_QFmaptype_derived_explicit_multiple_membersTscalar_and_array, i64 1, align 8
!CHECK: %[[MEMBER_ACCESS_1:.*]] = getelementptr %_QFmaptype_derived_explicit_multiple_membersTscalar_and_array, ptr %[[ALLOCA]], i32 0, i32 2
!CHECK: %[[MEMBER_ACCESS_2:.*]] = getelementptr %_QFmaptype_derived_explicit_multiple_membersTscalar_and_array, ptr %[[ALLOCA]], i32 0, i32 0
!CHECK: %[[ARR_END_OFF:.*]] = getelementptr i32, ptr %[[MEMBER_ACCESS_1]], i64 1
!CHECK: %[[ARR_END:.*]] = ptrtoint ptr %[[ARR_END_OFF]] to i64
!CHECK: %[[FIRST_MEMBER:.*]] = ptrtoint ptr %[[MEMBER_ACCESS_2]] to i64
!CHECK: %[[SIZE_DIFF:.*]] = sub i64 %[[ARR_END]], %[[FIRST_MEMBER]]
!CHECK: %[[SIZE:.*]] = sdiv exact i64 %[[SIZE_DIFF]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr %[[MEMBER_ACCESS_2]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_SIZE_ARR:.*]] = getelementptr inbounds [3 x i64], ptr %.offload_sizes, i32 0, i32 0
!CHECK: store i64 %[[SIZE]], ptr %[[OFFLOAD_SIZE_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR_2:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR_2]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR_2:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_ptrs, i32 0, i32 1
!CHECK: store ptr %[[MEMBER_ACCESS_1]], ptr %[[OFFLOAD_PTR_ARR_2]], align 8
!CHECK: %[[BASE_PTR_ARR_3:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR_3]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR_3:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_ptrs, i32 0, i32 2
!CHECK: store ptr %[[MEMBER_ACCESS_2]], ptr %[[OFFLOAD_PTR_ARR_3]], align 8

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_derived_explicit_member_with_bounds_{{.*}}
!CHECK: %[[ALLOCA:.*]] = alloca %_QFmaptype_derived_explicit_member_with_boundsTscalar_and_array, i64 1, align 8
!CHECK: %[[MEMBER_ACCESS:.*]] = getelementptr %_QFmaptype_derived_explicit_member_with_boundsTscalar_and_array, ptr %[[ALLOCA]], i32 0, i32 1
!CHECK: %[[ARR_OFF:.*]] = getelementptr inbounds [10 x i32], ptr %[[MEMBER_ACCESS]], i64 0, i64 1
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [1 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [1 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr %[[ARR_OFF]], ptr %[[OFFLOAD_PTR_ARR]], align 8

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_derived_explicit_nested_single_member_{{.*}}
!CHECK: %[[ALLOCA:.*]] = alloca %_QFmaptype_derived_explicit_nested_single_memberTscalar_and_array, i64 1, align 8
!CHECK: %[[MEMBER_ACCESS:.*]] = getelementptr %_QFmaptype_derived_explicit_nested_single_memberTscalar_and_array, ptr %[[ALLOCA]], i32 0, i32 2, i32 1
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [1 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr %[[MEMBER_ACCESS]], ptr %[[OFFLOAD_PTR_ARR]], align 8

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_derived_explicit_multiple_nested_members_{{.*}}
!CHECK: %[[ALLOCA:.*]] = alloca %_QFmaptype_derived_explicit_multiple_nested_membersTscalar_and_array, i64 1, align 8
!CHECK: %[[MEMBER_ACCESS_1:.*]] = getelementptr %_QFmaptype_derived_explicit_multiple_nested_membersTscalar_and_array, ptr %[[ALLOCA]], i32 0, i32 2, i32 0
!CHECK: %[[MEMBER_ACCESS_2:.*]] = getelementptr %_QFmaptype_derived_explicit_multiple_nested_membersTscalar_and_array, ptr %[[ALLOCA]], i32 0, i32 2, i32 1
!CHECK: %[[ARR_END_OFF:.*]] = getelementptr float, ptr %[[MEMBER_ACCESS_2]], i64 1
!CHECK: %[[ARR_END:.*]] = ptrtoint ptr %[[ARR_END_OFF]] to i64
!CHECK: %[[FIRST_MEMBER:.*]] = ptrtoint ptr %[[MEMBER_ACCESS_1]] to i64
!CHECK: %[[SIZE_DIFF:.*]] = sub i64 %[[ARR_END]], %[[FIRST_MEMBER]]
!CHECK: %[[SIZE:.*]] = sdiv exact i64 %[[SIZE_DIFF]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr %[[MEMBER_ACCESS_1]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_SIZE_ARR:.*]] = getelementptr inbounds [3 x i64], ptr %.offload_sizes, i32 0, i32 0
!CHECK: store i64 %[[SIZE]], ptr %[[OFFLOAD_SIZE_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR_2:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR_2]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR_2:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_ptrs, i32 0, i32 1
!CHECK: store ptr %[[MEMBER_ACCESS_1]], ptr %[[OFFLOAD_PTR_ARR_2]], align 8
!CHECK: %[[BASE_PTR_ARR_3:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR_3]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR_3:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_ptrs, i32 0, i32 2
!CHECK: store ptr %[[MEMBER_ACCESS_2]], ptr %[[OFFLOAD_PTR_ARR_3]], align 8

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_derived_explicit_nested_member_with_bounds_{{.*}}
!CHECK: %[[ALLOCA:.*]] = alloca %_QFmaptype_derived_explicit_nested_member_with_boundsTscalar_and_array, i64 1, align 8
!CHECK: %[[MEMBER_ACCESS:.*]] = getelementptr %_QFmaptype_derived_explicit_nested_member_with_boundsTscalar_and_array, ptr %[[ALLOCA]], i32 0, i32 2, i32 2
!CHECK: %[[ARR_OFF:.*]] = getelementptr inbounds [10 x i32], ptr %[[MEMBER_ACCESS]], i64 0, i64 1
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [1 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [1 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr %[[ARR_OFF]], ptr %[[OFFLOAD_PTR_ARR]], align 8

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_common_block_{{.*}}
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [1 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: store ptr @var_common_, ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [1 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr @var_common_, ptr %[[OFFLOAD_PTR_ARR]], align 8

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_common_block_members_{{.*}}
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: store ptr @var_common_, ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr @var_common_, ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR_1:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
!CHECK: store ptr getelementptr (i8, ptr @var_common_, i64 4), ptr %[[BASE_PTR_ARR_1]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR_1:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_ptrs, i32 0, i32 1
!CHECK: store ptr getelementptr (i8, ptr @var_common_, i64 4), ptr %[[OFFLOAD_PTR_ARR_1]], align 8
