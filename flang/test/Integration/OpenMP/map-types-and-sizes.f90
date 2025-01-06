!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!RUN: %flang_fc1 -emit-llvm -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -flang-deprecated-no-hlfir %s -o - | FileCheck %s

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

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [4 x i64] [i64 0, i64 24, i64 8, i64 4]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [4 x i64] [i64 32, i64 281474976711169, i64 281474976711171, i64 281474976711187]
subroutine mapType_ptr
  integer, pointer :: a
  !$omp target
     a = 10
  !$omp end target
end subroutine mapType_ptr

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [4 x i64] [i64 0, i64 24, i64 8, i64 4]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [4 x i64] [i64 32, i64 281474976711169, i64 281474976711171, i64 281474976711187]
subroutine mapType_allocatable
  integer, allocatable :: a
  allocate(a)
  !$omp target
     a = 10
  !$omp end target
  deallocate(a)
end subroutine mapType_allocatable

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [4 x i64] [i64 0, i64 24, i64 8, i64 4]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [4 x i64] [i64 32, i64 281474976710657, i64 281474976710659, i64 281474976710675]
subroutine mapType_ptr_explicit
  integer, pointer :: a
  !$omp target map(tofrom: a)
     a = 10
  !$omp end target
end subroutine mapType_ptr_explicit

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [4 x i64] [i64 0, i64 24, i64 8, i64 4]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [4 x i64] [i64 32, i64 281474976710657, i64 281474976710659, i64 281474976710675]
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

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [4 x i64] [i64 0, i64 48, i64 8, i64 0]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [4 x i64] [i64 32, i64 281474976710657, i64 281474976710659, i64 281474976710675]
subroutine mapType_derived_type_alloca()
  type :: one_layer
  real(4) :: i
  integer, allocatable :: scalar
  integer(4) :: array_i(10)
  real(4) :: j
  integer, allocatable :: array_j(:)
  integer(4) :: k
  end type one_layer

  type(one_layer) :: one_l

  allocate(one_l%array_j(10))

  !$omp target map(tofrom: one_l%array_j)
      one_l%array_j(1) = 10
  !$omp end target
end subroutine

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [8 x i64] [i64 0, i64 40, i64 8, i64 136, i64 48, i64 8, i64 0, i64 4]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [8 x i64] [i64 32, i64 281474976710657, i64 281474976710659, i64 281474976710675, i64 281474976710657, i64 281474976710659, i64 281474976710675, i64 281474976710659]
subroutine mapType_alloca_derived_type()
  type :: one_layer
  real(4) :: i
  integer, allocatable :: scalar
  integer(4) :: array_i(10)
  real(4) :: j
  integer, allocatable :: array_j(:)
  integer(4) :: k
  end type one_layer

  type(one_layer), allocatable :: one_l

  allocate(one_l)
  allocate(one_l%array_j(10))

  !$omp target map(tofrom: one_l%array_j, one_l%k)
      one_l%array_j(1) = 10
      one_l%k = 20
  !$omp end target
end subroutine

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [8 x i64] [i64 0, i64 40, i64 8, i64 240, i64 48, i64 8, i64 0, i64 4]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [8 x i64] [i64 32, i64 281474976710657, i64 281474976710659, i64 281474976710675, i64 281474976710657, i64 281474976710659, i64 281474976710675, i64 281474976710659]
subroutine mapType_alloca_nested_derived_type()
  type :: middle_layer
  real(4) :: i
  integer(4) :: array_i(10)
  integer, allocatable :: array_k(:)
  integer(4) :: k
  end type middle_layer

  type :: top_layer
  real(4) :: i
  integer, allocatable :: scalar
  integer(4) :: array_i(10)
  real(4) :: j
  integer, allocatable :: array_j(:)
  integer(4) :: k
  type(middle_layer) :: nest
  end type top_layer

  type(top_layer), allocatable :: one_l

  allocate(one_l)
  allocate(one_l%nest%array_k(10))

  !$omp target map(tofrom: one_l%nest%array_k, one_l%nest%k)
      one_l%nest%array_k(1) = 10
      one_l%nest%k = 20
  !$omp end target
end subroutine

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [4 x i64] [i64 0, i64 48, i64 8, i64 0]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [4 x i64] [i64 32, i64 281474976710657, i64 281474976710659, i64 281474976710675]
subroutine mapType_nested_derived_type_alloca()
  type :: middle_layer
  real(4) :: i
  integer(4) :: array_i(10)
  integer, allocatable :: array_k(:)
  integer(4) :: k
  end type middle_layer

  type :: top_layer
  real(4) :: i
  integer, allocatable :: scalar
  integer(4) :: array_i(10)
  real(4) :: j
  integer, allocatable :: array_j(:)
  integer(4) :: k
  type(middle_layer) :: nest
  end type top_layer

  type(top_layer) :: one_l

  allocate(one_l%nest%array_k(10))

  !$omp target map(tofrom: one_l%nest%array_k)
      one_l%nest%array_k(1) = 25
  !$omp end target
end subroutine

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [7 x i64] [i64 0, i64 64, i64 8, i64 0, i64 48, i64 8, i64 0]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [7 x i64] [i64 32, i64 281474976710657, i64 281474976710656, i64 281474976710672, i64 281474976710657, i64 281474976710659, i64 281474976710675]
subroutine mapType_nested_derived_type_member_idx()
type :: vertexes
  integer :: test
  integer(4), allocatable :: vertexx(:)
  integer(4), allocatable :: vertexy(:)
end type vertexes

type :: dtype
  real(4) :: i
  type(vertexes), allocatable :: vertexes(:)
  integer(4) :: array_i(10)
end type dtype

type(dtype) :: alloca_dtype

allocate(alloca_dtype%vertexes(4))
allocate(alloca_dtype%vertexes(2)%vertexy(10))

!$omp target map(tofrom: alloca_dtype%vertexes(2)%vertexy)
  alloca_dtype%vertexes(2)%vertexy(5) = 20
!$omp end target
end subroutine

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
!CHECK: %[[OFFLOAD_SIZE_ARR:.*]] = getelementptr inbounds [4 x i64], ptr %.offload_sizes, i32 0, i32 0
!CHECK: store i64 %[[DIV]], ptr %[[OFFLOAD_SIZE_ARR]], align 8

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_allocatable_explicit_{{.*}}
!CHECK: %[[ALLOCA:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8 }, i64 1, align 8
!CHECK: %[[ALLOCA_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8 }, ptr %[[ALLOCA]], i32 1
!CHECK: %[[ALLOCA_GEP_INT:.*]] = ptrtoint ptr %[[ALLOCA_GEP]] to i64
!CHECK: %[[ALLOCA_INT:.*]] = ptrtoint ptr %[[ALLOCA]] to i64
!CHECK: %[[SIZE_DIFF:.*]] = sub i64 %[[ALLOCA_GEP_INT]], %[[ALLOCA_INT]]
!CHECK: %[[DIV:.*]] = sdiv exact i64 %[[SIZE_DIFF]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
!CHECK: %[[OFFLOAD_SIZE_ARR:.*]] = getelementptr inbounds [4 x i64], ptr %.offload_sizes, i32 0, i32 0
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

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_derived_type_alloca_{{.*}}
!CHECK: %[[ALLOCATABLE_DESC_ALLOCA:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, align 8
!CHECK: %[[ALLOCA:.*]] = alloca %_QFmaptype_derived_type_allocaTone_layer, i64 1, align 8
!CHECK: %[[DESC_BOUND_ACCESS:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[ALLOCATABLE_DESC_ALLOCA]], i32 0, i32 7, i64 0, i32 1
!CHECK: %[[DESC_BOUND_ACCESS_LOAD:.*]] = load i64, ptr %[[DESC_BOUND_ACCESS]], align 8
!CHECK: %[[OFFSET_UB:.*]] = sub i64 %[[DESC_BOUND_ACCESS_LOAD]], 1
!CHECK: %[[MEMBER_ACCESS:.*]] = getelementptr %_QFmaptype_derived_type_allocaTone_layer, ptr %[[ALLOCA]], i32 0, i32 4
!CHECK: %[[MEMBER_DESCRIPTOR_BASE_ADDR:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[MEMBER_ACCESS]], i32 0, i32 0
!CHECK: %[[CALCULATE_DIM_SIZE:.*]] = sub i64 %[[OFFSET_UB]], 0
!CHECK: %[[RESTORE_OFFSET:.*]] = add i64 %[[CALCULATE_DIM_SIZE]], 1
!CHECK: %[[MEMBER_BASE_ADDR_SIZE:.*]] = mul i64 1, %[[RESTORE_OFFSET]]
!CHECK: %[[DESC_BASE_ADDR_DATA_SIZE:.*]] = mul i64 %[[MEMBER_BASE_ADDR_SIZE]], 4
!CHECK: %[[MEMBER_ACCESS_ADDR_END:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[MEMBER_ACCESS]], i64 1
!CHECK: %[[MEMBER_ACCESS_ADDR_INT:.*]] = ptrtoint ptr %[[MEMBER_ACCESS_ADDR_END]] to i64
!CHECK: %[[MEMBER_ACCESS_ADDR_BEGIN:.*]] = ptrtoint ptr %[[MEMBER_ACCESS]] to i64
!CHECK: %[[DTYPE_SEGMENT_SIZE:.*]] = sub i64 %[[MEMBER_ACCESS_ADDR_INT]], %[[MEMBER_ACCESS_ADDR_BEGIN]]
!CHECK: %[[DTYPE_SIZE_CALC:.*]] = sdiv exact i64 %[[DTYPE_SEGMENT_SIZE]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr %[[MEMBER_ACCESS]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_SIZE_ARR:.*]] = getelementptr inbounds [4 x i64], ptr %.offload_sizes, i32 0, i32 0
!CHECK: store i64 %[[DTYPE_SIZE_CALC]], ptr %[[OFFLOAD_SIZE_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 1
!CHECK: store ptr %[[MEMBER_ACCESS]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 2
!CHECK: store ptr %[[MEMBER_DESCRIPTOR_BASE_ADDR]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 3
!CHECK: store ptr %[[MEMBER_DESCRIPTOR_BASE_ADDR]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 3
!CHECK: store ptr %array_offset, ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_SIZE_ARR:.*]] = getelementptr inbounds [4 x i64], ptr %.offload_sizes, i32 0, i32 3
!CHECK: store i64 %[[DESC_BASE_ADDR_DATA_SIZE]], ptr %[[OFFLOAD_SIZE_ARR]], align 8

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_alloca_derived_type_{{.*}}
!CHECK: %{{.*}} = alloca { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, align 8
!CHECK: %[[DTYPE_DESC_ALLOCA:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, align 8
!CHECK: %[[DTYPE_ARRAY_MEMBER_DESC_ALLOCA:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, align 8
!CHECK: %[[DTYPE_DESC_ALLOCA_2:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, align 8
!CHECK: %[[DTYPE_DESC_ALLOCA_3:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, i64 1, align 8
!CHECK: %[[ACCESS_DESC_MEMBER_UB:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[DTYPE_ARRAY_MEMBER_DESC_ALLOCA]], i32 0, i32 7, i64 0, i32 1
!CHECK: %[[LOAD_DESC_MEMBER_UB:.*]] = load i64, ptr %[[ACCESS_DESC_MEMBER_UB]], align 8
!CHECK: %[[OFFSET_MEMBER_UB:.*]] = sub i64 %[[LOAD_DESC_MEMBER_UB]], 1
!CHECK: %[[DTYPE_BASE_ADDR_ACCESS:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[DTYPE_DESC_ALLOCA_2]], i32 0, i32 0
!CHECK: %[[DTYPE_BASE_ADDR_LOAD:.*]] = load ptr, ptr %[[DTYPE_BASE_ADDR_ACCESS]], align 8
!CHECK: %[[DTYPE_ALLOCA_MEMBER_ACCESS:.*]] = getelementptr %_QFmaptype_alloca_derived_typeTone_layer, ptr %[[DTYPE_BASE_ADDR_LOAD]], i32 0, i32 4
!CHECK: %[[DTYPE_ALLOCA_MEMBER_BASE_ADDR_ACCESS:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[DTYPE_ALLOCA_MEMBER_ACCESS]], i32 0, i32 0
!CHECK: %[[DTYPE_BASE_ADDR_ACCESS_2:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[DTYPE_DESC_ALLOCA]], i32 0, i32 0
!CHECK: %[[DTYPE_BASE_ADDR_LOAD_2:.*]] = load ptr, ptr %[[DTYPE_BASE_ADDR_ACCESS_2]], align 8
!CHECK: %[[DTYPE_NONALLOCA_MEMBER_ACCESS:.*]] = getelementptr %_QFmaptype_alloca_derived_typeTone_layer, ptr %[[DTYPE_BASE_ADDR_LOAD_2]], i32 0, i32 5
!CHECK: %[[DTYPE_BASE_ADDR_ACCESS_3:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[DTYPE_DESC_ALLOCA_3]], i32 0, i32 0
!CHECK: %[[MEMBER_SIZE_CALC_1:.*]] = sub i64 %[[OFFSET_MEMBER_UB]], 0
!CHECK: %[[MEMBER_SIZE_CALC_2:.*]] = add i64 %[[MEMBER_SIZE_CALC_1]], 1
!CHECK: %[[MEMBER_SIZE_CALC_3:.*]] = mul i64 1, %[[MEMBER_SIZE_CALC_2]]
!CHECK: %[[MEMBER_SIZE_CALC_4:.*]] = mul i64 %[[MEMBER_SIZE_CALC_3]], 4
!CHECK: %[[DTYPE_BASE_ADDR_LOAD_3:.*]] = load ptr, ptr %[[DTYPE_BASE_ADDR_ACCESS_3]], align 8
!CHECK: %[[LOAD_DTYPE_DESC_MEMBER:.*]] = load ptr, ptr %[[DTYPE_ALLOCA_MEMBER_BASE_ADDR_ACCESS]], align 8
!CHECK: %[[MEMBER_ARRAY_OFFSET:.*]] = getelementptr inbounds i32, ptr %[[LOAD_DTYPE_DESC_MEMBER]], i64 0
!CHECK: %[[DTYPE_END_OFFSET:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[DTYPE_DESC_ALLOCA_3]], i32 1
!CHECK: %[[DTYPE_END:.*]] = ptrtoint ptr %[[DTYPE_END_OFFSET]] to i64
!CHECK: %[[DTYPE_BEGIN:.*]] = ptrtoint ptr %[[DTYPE_DESC_ALLOCA_3]] to i64
!CHECK: %[[DTYPE_DESC_SZ_CALC:.*]] = sub i64 %[[DTYPE_END]], %[[DTYPE_BEGIN]]
!CHECK: %[[DTYPE_DESC_SZ:.*]] = sdiv exact i64 %[[DTYPE_DESC_SZ_CALC]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: store ptr %[[DTYPE_DESC_ALLOCA_3]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr %[[DTYPE_DESC_ALLOCA_3]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_SIZE_ARR:.*]] = getelementptr inbounds [8 x i64], ptr %.offload_sizes, i32 0, i32 0
!CHECK: store i64 %[[DTYPE_DESC_SZ]], ptr %[[OFFLOAD_SIZE_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
!CHECK: store ptr %[[DTYPE_DESC_ALLOCA_3]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_ptrs, i32 0, i32 1
!CHECK: store ptr %[[DTYPE_DESC_ALLOCA_3]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
!CHECK: store ptr %[[DTYPE_DESC_ALLOCA_3]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_ptrs, i32 0, i32 2
!CHECK: store ptr %[[DTYPE_BASE_ADDR_ACCESS_3]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_baseptrs, i32 0, i32 3
!CHECK: store ptr %[[DTYPE_BASE_ADDR_ACCESS_3]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_ptrs, i32 0, i32 3
!CHECK: store ptr %[[DTYPE_BASE_ADDR_LOAD_3]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_baseptrs, i32 0, i32 4
!CHECK: store ptr %[[DTYPE_DESC_ALLOCA_3]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_ptrs, i32 0, i32 4
!CHECK: store ptr %[[DTYPE_ALLOCA_MEMBER_ACCESS]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_baseptrs, i32 0, i32 5
!CHECK: store ptr %[[DTYPE_DESC_ALLOCA_3]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_ptrs, i32 0, i32 5
!CHECK: store ptr %[[DTYPE_ALLOCA_MEMBER_BASE_ADDR_ACCESS]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_baseptrs, i32 0, i32 6
!CHECK: store ptr %[[DTYPE_ALLOCA_MEMBER_BASE_ADDR_ACCESS]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_ptrs, i32 0, i32 6
!CHECK: store ptr %[[MEMBER_ARRAY_OFFSET]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_SIZE_ARR:.*]] = getelementptr inbounds [8 x i64], ptr %.offload_sizes, i32 0, i32 6
!CHECK: store i64 %[[MEMBER_SIZE_CALC_4]], ptr %[[OFFLOAD_SIZE_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_baseptrs, i32 0, i32 7
!CHECK: store ptr %[[DTYPE_DESC_ALLOCA_3]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_ptrs, i32 0, i32 7
!CHECK: store ptr %[[DTYPE_NONALLOCA_MEMBER_ACCESS]], ptr %[[OFFLOAD_PTR_ARR]], align 8

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_alloca_nested_derived_type{{.*}}
!CHECK: %{{.*}} = alloca { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, align 8
!CHECK: %[[DTYPE_DESC_ALLOCA_1:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, align 8
!CHECK: %[[ALLOCATABLE_MEMBER_ALLOCA:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, align 8
!CHECK: %[[DTYPE_DESC_ALLOCA_2:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, align 8
!CHECK: %[[DTYPE_DESC_ALLOCA_3:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, i64 1, align 8
!CHECK: %[[ALLOCATABLE_MEMBER_ALLOCA_UB:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[ALLOCATABLE_MEMBER_ALLOCA]], i32 0, i32 7, i64 0, i32 1
!CHECK: %[[ALLOCATABLE_MEMBER_ALLOCA_UB_LOAD:.*]] = load i64, ptr %[[ALLOCATABLE_MEMBER_ALLOCA_UB]], align 8
!CHECK: %[[ALLOCATABLE_MEMBER_SIZE_CALC_1:.*]] = sub i64 %[[ALLOCATABLE_MEMBER_ALLOCA_UB_LOAD]], 1
!CHECK: %[[DTYPE_DESC_BASE_ADDR_ACCESS:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[DTYPE_DESC_ALLOCA_2]], i32 0, i32 0
!CHECK: %[[DTYPE_DESC_BASE_ADDR_LOAD:.*]] = load ptr, ptr %[[DTYPE_DESC_BASE_ADDR_ACCESS]], align 8
!CHECK: %[[NESTED_DTYPE_ACCESS:.*]] = getelementptr %_QFmaptype_alloca_nested_derived_typeTtop_layer, ptr %[[DTYPE_DESC_BASE_ADDR_LOAD]], i32 0, i32 6
!CHECK: %[[MAPPED_MEMBER_ACCESS:.*]] = getelementptr %_QFmaptype_alloca_nested_derived_typeTmiddle_layer, ptr %[[NESTED_DTYPE_ACCESS]], i32 0, i32 2
!CHECK: %[[MAPPED_MEMBER_BASE_ADDR_ACCESS:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[MAPPED_MEMBER_ACCESS]], i32 0, i32 0
!CHECK: %[[DTYPE_DESC_BASE_ADDR_ACCESS_2:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[DTYPE_DESC_ALLOCA_1]], i32 0, i32 0
!CHECK: %[[DTYPE_DESC_BASE_ADDR_LOAD_2:.*]] = load ptr, ptr %[[DTYPE_DESC_BASE_ADDR_ACCESS_2]], align 8
!CHECK: %[[NESTED_DTYPE_ACCESS:.*]] = getelementptr %_QFmaptype_alloca_nested_derived_typeTtop_layer, ptr %[[DTYPE_DESC_BASE_ADDR_LOAD_2]], i32 0, i32 6
!CHECK: %[[NESTED_NONALLOCA_MEMBER_ACCESS:.*]] = getelementptr %_QFmaptype_alloca_nested_derived_typeTmiddle_layer, ptr %[[NESTED_DTYPE_ACCESS]], i32 0, i32 3
!CHECK: %[[DTYPE_DESC_BASE_ADDR:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[DTYPE_DESC_ALLOCA_3]], i32 0, i32 0
!CHECK: %[[ALLOCATABLE_MEMBER_SIZE_CALC_2:.*]] = sub i64 %[[ALLOCATABLE_MEMBER_SIZE_CALC_1]], 0
!CHECK: %[[ALLOCATABLE_MEMBER_SIZE_CALC_3:.*]] = add i64 %[[ALLOCATABLE_MEMBER_SIZE_CALC_2]], 1
!CHECK: %[[ALLOCATABLE_MEMBER_SIZE_CALC_4:.*]] = mul i64 1, %[[ALLOCATABLE_MEMBER_SIZE_CALC_3]]
!CHECK: %[[ALLOCATABLE_MEMBER_SIZE_CALC_5:.*]] = mul i64 %[[ALLOCATABLE_MEMBER_SIZE_CALC_4]], 4
!CHECK: %[[LOAD_BASE_ADDR:.*]] = load ptr, ptr %[[DTYPE_DESC_BASE_ADDR]], align 8
!CHECK: %[[LOAD_DESC_MEMBER_BASE_ADDR:.*]] = load ptr, ptr %[[MAPPED_MEMBER_BASE_ADDR_ACCESS]], align 8
!CHECK: %[[ARRAY_OFFSET:.*]] = getelementptr inbounds i32, ptr %[[LOAD_DESC_MEMBER_BASE_ADDR]], i64 0
!CHECK: %[[DTYPE_DESC_SIZE_CALC_1:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[DTYPE_DESC_ALLOCA_3]], i32 1
!CHECK: %[[DTYPE_DESC_SIZE_CALC_2:.*]] = ptrtoint ptr %[[DTYPE_DESC_SIZE_CALC_1]] to i64
!CHECK: %[[DTYPE_DESC_SIZE_CALC_3:.*]] = ptrtoint ptr %[[DTYPE_DESC_ALLOCA_3]] to i64
!CHECK: %[[DTYPE_DESC_SIZE_CALC_4:.*]] = sub i64 %[[DTYPE_DESC_SIZE_CALC_2]], %[[DTYPE_DESC_SIZE_CALC_3]]
!CHECK: %[[DTYPE_DESC_SIZE_CALC_5:.*]] = sdiv exact i64 %[[DTYPE_DESC_SIZE_CALC_4]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: store ptr %[[DTYPE_DESC_ALLOCA_3]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr %[[DTYPE_DESC_ALLOCA_3]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_SIZE_ARR:.*]] = getelementptr inbounds [8 x i64], ptr %.offload_sizes, i32 0, i32 0
!CHECK: store i64 %[[DTYPE_DESC_SIZE_CALC_5]], ptr %[[OFFLOAD_SIZE_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
!CHECK: store ptr %[[DTYPE_DESC_ALLOCA_3]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_ptrs, i32 0, i32 1
!CHECK: store ptr %[[DTYPE_DESC_ALLOCA_3]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
!CHECK: store ptr %[[DTYPE_DESC_ALLOCA_3]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_ptrs, i32 0, i32 2
!CHECK: store ptr %[[DTYPE_DESC_BASE_ADDR]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_baseptrs, i32 0, i32 3
!CHECK: store ptr %[[DTYPE_DESC_BASE_ADDR]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_ptrs, i32 0, i32 3
!CHECK: store ptr %[[LOAD_BASE_ADDR]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_baseptrs, i32 0, i32 4
!CHECK: store ptr %[[DTYPE_DESC_ALLOCA_3]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_ptrs, i32 0, i32 4
!CHECK: store ptr %[[MAPPED_MEMBER_ACCESS]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_baseptrs, i32 0, i32 5
!CHECK: store ptr %[[DTYPE_DESC_ALLOCA_3]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_ptrs, i32 0, i32 5
!CHECK: store ptr %[[MAPPED_MEMBER_BASE_ADDR_ACCESS]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_baseptrs, i32 0, i32 6
!CHECK: store ptr %[[MAPPED_MEMBER_BASE_ADDR_ACCESS]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_ptrs, i32 0, i32 6
!CHECK: store ptr %[[ARRAY_OFFSET]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_SIZE_ARR:.*]] = getelementptr inbounds [8 x i64], ptr %.offload_sizes, i32 0, i32 6
!CHECK: store i64 %[[ALLOCATABLE_MEMBER_SIZE_CALC_5]], ptr %[[OFFLOAD_SIZE_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_baseptrs, i32 0, i32 7
!CHECK: store ptr %[[DTYPE_DESC_ALLOCA_3]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_ptrs, i32 0, i32 7
!CHECK: store ptr %[[NESTED_NONALLOCA_MEMBER_ACCESS]], ptr %[[OFFLOAD_PTR_ARR]], align 8

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_nested_derived_type_alloca{{.*}}
!CHECK: %[[ALLOCATABLE_MEMBER_ALLOCA:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, align 8
!CHECK: %[[ALLOCA:.*]] = alloca %_QFmaptype_nested_derived_type_allocaTtop_layer, i64 1, align 8
!CHECK: %[[ALLOCATABLE_MEMBER_BASE_ADDR:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[ALLOCATABLE_MEMBER_ALLOCA]], i32 0, i32 7, i64 0, i32 1
!CHECK: %[[ALLOCATABLE_MEMBER_ADDR_LOAD:.*]] = load i64, ptr %[[ALLOCATABLE_MEMBER_BASE_ADDR]], align 8
!CHECK: %[[ALLOCATABLE_MEMBER_SIZE_CALC_1:.*]] = sub i64 %[[ALLOCATABLE_MEMBER_ADDR_LOAD]], 1
!CHECK: %[[NESTED_DTYPE_MEMBER_ACCESS:.*]] = getelementptr %_QFmaptype_nested_derived_type_allocaTtop_layer, ptr %[[ALLOCA]], i32 0, i32 6
!CHECK: %[[NESTED_MEMBER_ACCESS:.*]] = getelementptr %_QFmaptype_nested_derived_type_allocaTmiddle_layer, ptr %[[NESTED_DTYPE_MEMBER_ACCESS]], i32 0, i32 2
!CHECK: %[[NESTED_MEMBER_BASE_ADDR_ACCESS:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %{{.*}}, i32 0, i32 0
!CHECK: %[[ALLOCATABLE_MEMBER_SIZE_CALC_2:.*]] = sub i64 %[[ALLOCATABLE_MEMBER_SIZE_CALC_1]], 0
!CHECK: %[[ALLOCATABLE_MEMBER_SIZE_CALC_3:.*]] = add i64 %[[ALLOCATABLE_MEMBER_SIZE_CALC_2]], 1
!CHECK: %[[ALLOCATABLE_MEMBER_SIZE_CALC_4:.*]] = mul i64 1, %[[ALLOCATABLE_MEMBER_SIZE_CALC_3]]
!CHECK: %[[ALLOCATABLE_MEMBER_SIZE_CALC_5:.*]] = mul i64 %[[ALLOCATABLE_MEMBER_SIZE_CALC_4]], 4
!CHECK: %[[LOAD_BASE_ADDR:.*]] = load ptr, ptr %[[NESTED_MEMBER_BASE_ADDR_ACCESS]], align 8
!CHECK: %[[ARR_OFFS:.*]] = getelementptr inbounds i32, ptr %[[LOAD_BASE_ADDR]], i64 0
!CHECK: %[[NESTED_MEMBER_BASE_ADDR_ACCESS_2:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[NESTED_MEMBER_ACCESS]], i64 1
!CHECK: %[[DTYPE_SEGMENT_SIZE_CALC_1:.*]] = ptrtoint ptr %[[NESTED_MEMBER_BASE_ADDR_ACCESS_2]] to i64
!CHECK: %[[DTYPE_SEGMENT_SIZE_CALC_2:.*]] = ptrtoint ptr %[[NESTED_MEMBER_ACCESS]] to i64
!CHECK: %[[DTYPE_SEGMENT_SIZE_CALC_3:.*]] = sub i64 %[[DTYPE_SEGMENT_SIZE_CALC_1]], %[[DTYPE_SEGMENT_SIZE_CALC_2]]
!CHECK: %[[DTYPE_SEGMENT_SIZE_CALC_4:.*]] = sdiv exact i64 %[[DTYPE_SEGMENT_SIZE_CALC_3]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr %[[NESTED_MEMBER_ACCESS]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_SIZE_ARR:.*]] = getelementptr inbounds [4 x i64], ptr %.offload_sizes, i32 0, i32 0
!CHECK: store i64 %[[DTYPE_SEGMENT_SIZE_CALC_4]], ptr %[[OFFLOAD_SIZE_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 1
!CHECK: store ptr %[[NESTED_MEMBER_ACCESS]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 2
!CHECK: store ptr %[[NESTED_MEMBER_BASE_ADDR_ACCESS]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 3
!CHECK: store ptr %[[NESTED_MEMBER_BASE_ADDR_ACCESS]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 3
!CHECK: store ptr %[[ARR_OFFS]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_SIZE_ARR:.*]] = getelementptr inbounds [4 x i64], ptr %.offload_sizes, i32 0, i32 3
!CHECK: store i64 %[[ALLOCATABLE_MEMBER_SIZE_CALC_5]], ptr %[[OFFLOAD_SIZE_ARR]], align 8

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_nested_derived_type_member_idx{{.*}}
!CHECK: %[[ALLOCA_0:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, align 8
!CHECK: %[[ALLOCA_1:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, align 8
!CHECK: %[[ALLOCA:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, align 8
!CHECK: %[[BASE_PTR_1:.*]] = alloca %_QFmaptype_nested_derived_type_member_idxTdtype, i64 1, align 8
!CHECK: %{{.*}} = getelementptr %_QFmaptype_nested_derived_type_member_idxTdtype, ptr %[[BASE_PTR_1]], i32 0, i32 1
!CHECK: %[[BOUNDS_ACC:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %[[ALLOCA]], i32 0, i32 7, i64 0, i32 1
!CHECK: %[[BOUNDS_LD:.*]] = load i64, ptr %[[BOUNDS_ACC]], align 8
!CHECK: %[[BOUNDS_ACC_2:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[ALLOCA_1]], i32 0, i32 7, i64 0, i32 1
!CHECK: %[[BOUNDS_LD_2:.*]] = load i64, ptr %[[BOUNDS_ACC_2]], align 8
!CHECK: %[[BOUNDS_CALC:.*]] = sub i64 %[[BOUNDS_LD_2]], 1
!CHECK: %[[OFF_PTR_1:.*]] = getelementptr %_QFmaptype_nested_derived_type_member_idxTdtype, ptr %[[BASE_PTR_1]], i32 0, i32 1
!CHECK: %[[OFF_PTR_CALC_0:.*]] = sub i64 %[[BOUNDS_LD]], 1
!CHECK: %[[OFF_PTR_2:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %[[OFF_PTR_1]], i32 0, i32 0
!CHECK: %[[GEP_DESC_PTR:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %[[ALLOCA_0]], i32 0, i32 0
!CHECK: %[[LOAD_DESC_PTR:.*]] = load ptr, ptr %[[GEP_DESC_PTR]], align 8
!CHECK: %[[SZ_CALC_1:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %[[ALLOCA_0]], i32 0, i32 7, i32 0, i32 2
!CHECK: %[[SZ_CALC_2:.*]] = load i64, ptr %[[SZ_CALC_1]], align 8
!CHECK: %[[SZ_CALC_3:.*]] = mul nsw i64 1, %[[SZ_CALC_2]]
!CHECK: %[[SZ_CALC_4:.*]] = add nsw i64 %[[SZ_CALC_3]], 0
!CHECK: %[[SZ_CALC_5:.*]] = getelementptr i8, ptr %[[LOAD_DESC_PTR]], i64 %[[SZ_CALC_4]]
!CHECK: %[[SZ_CALC_6:.*]] = getelementptr %_QFmaptype_nested_derived_type_member_idxTvertexes, ptr %[[SZ_CALC_5]], i32 0, i32 2
!CHECK: %[[OFF_PTR_4:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[SZ_CALC_6]], i32 0, i32 0
!CHECK: %[[OFF_PTR_CALC_1:.*]] = sub i64 %[[OFF_PTR_CALC_0]], 0
!CHECK: %[[OFF_PTR_CALC_2:.*]] = add i64 %[[OFF_PTR_CALC_1]], 1
!CHECK: %[[OFF_PTR_CALC_3:.*]] = mul i64 1, %[[OFF_PTR_CALC_2]]
!CHECK: %[[OFF_PTR_3:.*]] = mul i64 %[[OFF_PTR_CALC_3]], 104
!CHECK: %[[SZ_CALC_1_2:.*]] = sub i64 %[[BOUNDS_CALC]], 0
!CHECK: %[[SZ_CALC_2_2:.*]] = add i64 %[[SZ_CALC_1_2]], 1
!CHECK: %[[SZ_CALC_3_2:.*]] = mul i64 1, %[[SZ_CALC_2_2]]
!CHECK: %[[SZ_CALC_4_2:.*]] = mul i64 %[[SZ_CALC_3_2]], 4
!CHECK: %[[LOAD_OFF_PTR:.*]] = load ptr, ptr %[[OFF_PTR_2]], align 8
!CHECK: %[[ARR_OFFS:.*]] = getelementptr inbounds %_QFmaptype_nested_derived_type_member_idxTvertexes, ptr %[[LOAD_OFF_PTR]], i64 0
!CHECK: %[[LOAD_ARR_OFFS:.*]] = load ptr, ptr %[[OFF_PTR_4]], align 8
!CHECK: %[[ARR_OFFS_1:.*]] = getelementptr inbounds i32, ptr %[[LOAD_ARR_OFFS]], i64 0
!CHECK: %[[SZ_CALC_1:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %[[OFF_PTR_1]], i64 1
!CHECK: %[[SZ_CALC_2:.*]] = ptrtoint ptr %[[SZ_CALC_1]] to i64
!CHECK: %[[SZ_CALC_3:.*]] = ptrtoint ptr %[[OFF_PTR_1]] to i64
!CHECK: %[[SZ_CALC_4:.*]] = sub i64 %[[SZ_CALC_2]], %[[SZ_CALC_3]]
!CHECK: %[[SZ_CALC_5:.*]] = sdiv exact i64 %[[SZ_CALC_4]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: store ptr %[[BASE_PTR_1]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr %[[OFF_PTR_1]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_SIZE_ARR:.*]] = getelementptr inbounds [7 x i64], ptr %.offload_sizes, i32 0, i32 0
!CHECK: store i64 %[[SZ_CALC_5]], ptr %[[OFFLOAD_SIZE_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
!CHECK: store ptr %[[BASE_PTR_1]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 1
!CHECK: store ptr %[[OFF_PTR_1]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
!CHECK: store ptr %[[BASE_PTR_1]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 2
!CHECK: store ptr %[[OFF_PTR_2]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 3
!CHECK: store ptr %[[OFF_PTR_2]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 3
!CHECK: store ptr %[[ARR_OFFS]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_SIZE_ARR:.*]] = getelementptr inbounds [7 x i64], ptr %.offload_sizes, i32 0, i32 3
!CHECK: store i64 %[[OFF_PTR_3]], ptr %[[OFFLOAD_SIZE_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 4
!CHECK: store ptr %[[BASE_PTR_1]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 4
!CHECK: store ptr %[[SZ_CALC_6]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 5
!CHECK: store ptr %[[BASE_PTR_1]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 5
!CHECK: store ptr %[[OFF_PTR_4]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 6
!CHECK: store ptr %[[OFF_PTR_4]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 6
!CHECK: store ptr %[[ARR_OFFS_1]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_SIZE_ARR:.*]] = getelementptr inbounds [7 x i64], ptr %.offload_sizes, i32 0, i32 6
!CHECK: store i64 %[[SZ_CALC_4_2]], ptr %[[OFFLOAD_SIZE_ARR]], align 8

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
