!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!RUN: %flang_fc1 -emit-llvm -fopenmp -mmlir --enable-delayed-privatization-staging=false -fopenmp-version=61 -fopenmp-targets=amdgcn-amd-amdhsa %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NO-FPRIV
!RUN: %flang_fc1 -emit-llvm -fopenmp -mmlir --enable-delayed-privatization-staging=true -fopenmp-version=61 -fopenmp-targets=amdgcn-amd-amdhsa %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-FPRIV


!===============================================================================
! Check MapTypes for target implicit captures
!===============================================================================

!CHECK: @.offload_sizes = private unnamed_addr constant [2 x i64] [i64 4, i64 0]
!CHECK-FPRIV: @.offload_maptypes = private unnamed_addr constant [2 x i64] [i64 289, i64 288]
!CHECK-NO-FPRIV: @.offload_maptypes = private unnamed_addr constant [2 x i64] [i64 800, i64 288]
subroutine mapType_scalar
  integer :: a
  !$omp target
     a = 10
  !$omp end target
end subroutine mapType_scalar

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [2 x i64] [i64 4096, i64 0]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [2 x i64] [i64 547, i64 288]
subroutine mapType_array
  integer :: a(1024)
  !$omp target
     a(10) = 20
  !$omp end target
end subroutine mapType_array

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [2 x i64] [i64 8, i64 0]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [2 x i64] [i64 33, i64 288]
subroutine mapType_is_device_ptr
  use iso_c_binding, only : c_ptr
  type(c_ptr) :: p
  !$omp target is_device_ptr(p)
  !$omp end target
end subroutine mapType_is_device_ptr

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [5 x i64] [i64 0, i64 24, i64 0, i64 0, i64 0]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [5 x i64] [i64 545, i64 281474976711173, i64 515, i64 16384, i64 288]
subroutine mapType_ptr
  integer, pointer :: a
  !$omp target
     a = 10
  !$omp end target
end subroutine mapType_ptr

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [1 x i64] [i64 4]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [1 x i64] [i64 4097]
subroutine map_present_target_data
  integer :: x
!$omp target data map(present, to: x)
!$omp end target data
end subroutine

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [1 x i64] [i64 4]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [1 x i64] [i64 4097]
subroutine map_present_update
  integer :: x
!$omp target update to(present: x)
end subroutine

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [1 x i64] [i64 4]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [1 x i64] [i64 1027]
subroutine map_close
  integer :: x
!$omp target data map(close, tofrom: x)
!$omp end target data
end subroutine

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [1 x i64] [i64 4]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [1 x i64] [i64 8195]
subroutine map_ompx_hold
  integer :: x
!$omp target data map(ompx_hold, tofrom: x)
!$omp end target data
end subroutine

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [5 x i64] [i64 0, i64 24, i64 0, i64 0, i64 0]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [5 x i64] [i64 545, i64 281474976711173, i64 515, i64 16384, i64 288]
subroutine mapType_allocatable
  integer, allocatable :: a
  allocate(a)
  !$omp target
     a = 10
  !$omp end target
  deallocate(a)
end subroutine mapType_allocatable

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [5 x i64] [i64 0, i64 24, i64 0, i64 0, i64 0]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [5 x i64] [i64 33, i64 281474976710661, i64 3, i64 16384, i64 288]
subroutine mapType_ptr_explicit
  integer, pointer :: a
  !$omp target map(tofrom: a)
     a = 10
  !$omp end target
end subroutine mapType_ptr_explicit

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [5 x i64] [i64 0, i64 24, i64 0, i64 0, i64 0]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [5 x i64] [i64 33, i64 281474976710661, i64 3, i64 16384, i64 288]
subroutine mapType_allocatable_explicit
  integer, allocatable :: a
  allocate(a)
  !$omp target map(tofrom: a)
     a = 10
  !$omp end target
  deallocate(a)
end subroutine mapType_allocatable_explicit

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [2 x i64] [i64 48, i64 0]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [2 x i64] [i64 547, i64 288]
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

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [2 x i64] [i64 48, i64 0]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [2 x i64] [i64 35, i64 288]
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

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [2 x i64] [i64 40, i64 0]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [2 x i64] [i64 35, i64 288]
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

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [4 x i64] [i64 0, i64 4, i64 4, i64 0]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [4 x i64] [i64 32, i64 281474976710659, i64 281474976710659, i64 288]
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

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [2 x i64] [i64 16, i64 0]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [2 x i64] [i64 35, i64 288]
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

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [2 x i64] [i64 4, i64 0]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [2 x i64] [i64 35, i64 288]
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

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [4 x i64] [i64 0, i64 4, i64 4, i64 0]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [4 x i64] [i64 32, i64 281474976710659, i64 281474976710659, i64 288]
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

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [2 x i64] [i64 16, i64 0]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [2 x i64] [i64 35, i64 288]
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

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [5 x i64] [i64 0, i64 48, i64 0, i64 0, i64 0]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [5 x i64] [i64 32, i64 281474976710661, i64 3, i64 16384, i64 288]
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

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [9 x i64] [i64 0, i64 40, i64 0, i64 48, i64 0, i64 4, i64 0, i64 0, i64 0]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [9 x i64] [i64 33, i64 281474976710661, i64 0, i64 281474976710661, i64 3, i64 281474976710659, i64 16384, i64 16384, i64 288]
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

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [9 x i64] [i64 0, i64 40, i64 0, i64 48, i64 0, i64 4, i64 0, i64 0, i64 0]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [9 x i64] [i64 33, i64 281474976710661, i64 0, i64 281474976710661, i64 3, i64 281474976710659, i64 16384, i64 16384, i64 288]
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

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [5 x i64] [i64 0, i64 48, i64 0, i64 0, i64 0]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [5 x i64] [i64 32, i64 281474976710661, i64 3, i64 16384, i64 288]
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

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [8 x i64] [i64 0, i64 64, i64 0, i64 48, i64 0, i64 0, i64 0, i64 0]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [8 x i64] [i64 32, i64 281474976710661, i64 0, i64 281474976710661, i64 3, i64 16384, i64 16384, i64 288]
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

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [3 x i64] [i64 8, i64 4, i64 0]
!CHECK-FPRIV: @.offload_maptypes{{.*}} = private unnamed_addr constant [3 x i64] [i64 544, i64 289, i64 288]
!CHECK-NO-FPRIV: @.offload_maptypes{{.*}} = private unnamed_addr constant [3 x i64] [i64 544, i64 800, i64 288]
subroutine mapType_c_ptr
  use iso_c_binding, only : c_ptr, c_loc
  type(c_ptr) :: a
  integer, target :: b
  !$omp target
     a = c_loc(b)
  !$omp end target
end subroutine mapType_c_ptr

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [2 x i64] [i64 1, i64 0]
!CHECK-FPRIV: @.offload_maptypes{{.*}} = private unnamed_addr constant [2 x i64] [i64 289, i64 288]
!CHECK-NO-FPRIV: @.offload_maptypes{{.*}} = private unnamed_addr constant [2 x i64] [i64 800, i64 288]
subroutine mapType_char
  character :: a
  !$omp target
     a = 'b'
  !$omp end target
end subroutine mapType_char

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [2 x i64] [i64 8, i64 0]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [2 x i64] [i64 35, i64 288]
subroutine mapType_common_block
  implicit none
  common /var_common/ var1, var2
  integer :: var1, var2
!$omp target map(tofrom: /var_common/)
  var1 = var1 + 20
  var2 = var2 + 30
!$omp end target
end subroutine mapType_common_block

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [3 x i64] [i64 4, i64 4, i64 0]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [3 x i64] [i64 35, i64 35, i64 288]
subroutine mapType_common_block_members
  implicit none
  common /var_common/ var1, var2
  integer :: var1, var2

!$omp target map(tofrom: var1, var2)
  var2 = var1
!$omp end target
end subroutine mapType_common_block_members

!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [7 x i64] [i64 0, i64 40, i64 0, i64 4, i64 0, i64 0, i64 0]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [7 x i64] [i64 33, i64 281474976710661, i64 3, i64 35, i64 16384, i64 16384, i64 288]
!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [7 x i64] [i64 4, i64 0, i64 40, i64 0, i64 0, i64 0, i64 0]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [7 x i64] [i64 35, i64 545, i64 562949953421829, i64 515, i64 16384, i64 16384, i64 288]
!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [3 x i64] [i64 4, i64 48, i64 0]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [3 x i64] [i64 35, i64 547, i64 288]
!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [7 x i64] [i64 0, i64 64, i64 0, i64 4, i64 0, i64 0, i64 0]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [7 x i64] [i64 33, i64 281474976710661, i64 3, i64 35, i64 16384, i64 16384, i64 288]
!CHECK: @.offload_sizes{{.*}} = private unnamed_addr constant [7 x i64] [i64 0, i64 64, i64 0, i64 4, i64 0, i64 0, i64 0]
!CHECK: @.offload_maptypes{{.*}} = private unnamed_addr constant [7 x i64] [i64 33, i64 281474976710661, i64 3, i64 35, i64 16384, i64 16384, i64 288]

module maptype_polymorphic_mod
  type :: base_t
    integer :: base
  end type

  type, extends(base_t) :: child_t
    integer :: child
  end type

  type :: wrapper_t
    integer :: marker
    class(base_t), allocatable :: item
  end type
end module

subroutine mapType_polymorphic_allocatable_explicit(obj, res)
  use maptype_polymorphic_mod
  class(base_t), allocatable :: obj
  logical :: res

!$omp target map(tofrom: obj, res)
  obj%base = obj%base + 1
  res = .true.
!$omp end target
end subroutine mapType_polymorphic_allocatable_explicit

subroutine mapType_polymorphic_allocatable_implicit(obj, res)
  use maptype_polymorphic_mod
  class(base_t), allocatable :: obj
  logical :: res

!$omp target map(tofrom: res)
  obj%base = obj%base + 1
  res = .true.
!$omp end target
end subroutine mapType_polymorphic_allocatable_implicit

subroutine mapType_polymorphic_nested_implicit(wrapper, res)
  use maptype_polymorphic_mod
  type(wrapper_t) :: wrapper
  logical :: res

!$omp target map(tofrom: res)
  wrapper%item%base = wrapper%item%base + 1
  wrapper%marker = wrapper%marker + 1
  res = .true.
!$omp end target
end subroutine mapType_polymorphic_nested_implicit

subroutine mapType_polymorphic_array_section_explicit(obj, res)
  use maptype_polymorphic_mod
  class(base_t), allocatable :: obj(:)
  integer :: res

!$omp target map(tofrom: obj(2:5), res)
  obj(2)%base = obj(2)%base + 1
  res = 1
!$omp end target
end subroutine mapType_polymorphic_array_section_explicit

subroutine mapType_polymorphic_array_whole_explicit(obj, res)
  use maptype_polymorphic_mod
  class(base_t), allocatable :: obj(:)
  integer :: res

!$omp target map(tofrom: obj, res)
  obj(1)%base = obj(1)%base + 1
  res = 1
!$omp end target
end subroutine mapType_polymorphic_array_whole_explicit

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_ptr_explicit_{{.*}}
!CHECK: %[[ALLOCA:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8 }, i64 1, align 8
!CHECK: %[[ALLOCA_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8 }, ptr %[[ALLOCA]], i32 0, i32 0
!CHECK: %[[LOAD_PTR:.*]] = load ptr, ptr %[[ALLOCA_GEP]], align 8
!CHECK: %[[LOAD_PTR2:.*]] = load ptr, ptr %[[ALLOCA_GEP]], align 8
!CHECK: %[[NULL_CMP:.*]] = icmp eq ptr %[[LOAD_PTR2]], null
!CHECK: %[[SELECT:.*]] = select i1 %[[NULL_CMP]], i64 0, i64 4
!CHECK: %[[OFFLOAD_SIZE_ARR:.*]] = getelementptr inbounds [5 x i64], ptr %.offload_sizes, i32 0, i32 2
!CHECK: store i64 %[[SELECT]], ptr %[[OFFLOAD_SIZE_ARR]], align 8

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_allocatable_explicit_{{.*}}
!CHECK: %[[ALLOCA:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8 }, i64 1, align 8
!CHECK: %[[ALLOCA_GEP:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8 }, ptr %[[ALLOCA]], i32 0, i32 0
!CHECK: %[[LOAD_PTR:.*]] = load ptr, ptr %[[ALLOCA_GEP]], align 8
!CHECK: %[[LOAD_PTR2:.*]] = load ptr, ptr %[[ALLOCA_GEP]], align 8
!CHECK: %[[NULL_CMP:.*]] = icmp eq ptr %[[LOAD_PTR2]], null
!CHECK: %[[SELECT:.*]] = select i1 %[[NULL_CMP]], i64 0, i64 4
!CHECK: %[[OFFLOAD_SIZE_ARR:.*]] = getelementptr inbounds [5 x i64], ptr %.offload_sizes, i32 0, i32 2
!CHECK: store i64 %[[SELECT]], ptr %[[OFFLOAD_SIZE_ARR]], align 8

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_derived_implicit_{{.*}}
!CHECK: %[[ALLOCA:.*]] = alloca %_QFmaptype_derived_implicitTscalar_and_array, i64 1, align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr %[[ALLOCA]], ptr %[[OFFLOAD_PTR_ARR]], align 8

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_derived_explicit_{{.*}}
!CHECK: %[[ALLOCA:.*]] = alloca %_QFmaptype_derived_explicitTscalar_and_array, i64 1, align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr %[[ALLOCA]], ptr %[[OFFLOAD_PTR_ARR]], align 8

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_derived_explicit_single_member_{{.*}}
!CHECK: %[[ALLOCA:.*]] = alloca %_QFmaptype_derived_explicit_single_memberTscalar_and_array, i64 1, align 8
!CHECK: %[[MEMBER_ACCESS:.*]] = getelementptr %_QFmaptype_derived_explicit_single_memberTscalar_and_array, ptr %[[ALLOCA]], i32 0, i32 1
!CHECK: %[[ARR_OFF:.*]] = getelementptr inbounds [10 x i32], ptr %[[MEMBER_ACCESS]], i64 0, i64 0
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr %[[ARR_OFF]], ptr %[[OFFLOAD_PTR_ARR]], align 8

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_derived_explicit_multiple_members_{{.*}}
!CHECK: %[[ALLOCA:.*]] = alloca %_QFmaptype_derived_explicit_multiple_membersTscalar_and_array, i64 1, align 8
!CHECK: %[[MEMBER_ACCESS_1:.*]] = getelementptr %_QFmaptype_derived_explicit_multiple_membersTscalar_and_array, ptr %[[ALLOCA]], i32 0, i32 2
!CHECK: %[[MEMBER_ACCESS_2:.*]] = getelementptr %_QFmaptype_derived_explicit_multiple_membersTscalar_and_array, ptr %[[ALLOCA]], i32 0, i32 0
!CHECK: %[[ARR_END_OFF:.*]] = getelementptr i32, ptr %[[MEMBER_ACCESS_1]], i64 1
!CHECK: %[[ARR_END:.*]] = ptrtoaddr ptr %[[ARR_END_OFF]] to i64
!CHECK: %[[FIRST_MEMBER:.*]] = ptrtoaddr ptr %[[MEMBER_ACCESS_2]] to i64
!CHECK: %[[SIZE:.*]] = sub i64 %[[ARR_END]], %[[FIRST_MEMBER]]
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr %[[MEMBER_ACCESS_2]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_SIZE_ARR:.*]] = getelementptr inbounds [4 x i64], ptr %.offload_sizes, i32 0, i32 0
!CHECK: store i64 %[[SIZE]], ptr %[[OFFLOAD_SIZE_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR_2:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR_2]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR_2:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 1
!CHECK: store ptr %[[MEMBER_ACCESS_1]], ptr %[[OFFLOAD_PTR_ARR_2]], align 8
!CHECK: %[[BASE_PTR_ARR_3:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR_3]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR_3:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 2
!CHECK: store ptr %[[MEMBER_ACCESS_2]], ptr %[[OFFLOAD_PTR_ARR_3]], align 8

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_derived_explicit_member_with_bounds_{{.*}}
!CHECK: %[[ALLOCA:.*]] = alloca %_QFmaptype_derived_explicit_member_with_boundsTscalar_and_array, i64 1, align 8
!CHECK: %[[MEMBER_ACCESS:.*]] = getelementptr %_QFmaptype_derived_explicit_member_with_boundsTscalar_and_array, ptr %[[ALLOCA]], i32 0, i32 1
!CHECK: %[[ARR_OFF:.*]] = getelementptr inbounds [10 x i32], ptr %[[MEMBER_ACCESS]], i64 0, i64 1
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr %[[ARR_OFF]], ptr %[[OFFLOAD_PTR_ARR]], align 8

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_derived_explicit_nested_single_member_{{.*}}
!CHECK: %[[ALLOCA:.*]] = alloca %_QFmaptype_derived_explicit_nested_single_memberTscalar_and_array, i64 1, align 8
!CHECK: %[[MEMBER_ACCESS:.*]] = getelementptr %_QFmaptype_derived_explicit_nested_single_memberTscalar_and_array, ptr %[[ALLOCA]], i32 0, i32 2
!CHECK: %[[MEMBER_ACCESS_2:.*]] = getelementptr %_QFmaptype_derived_explicit_nested_single_memberTnested, ptr %[[MEMBER_ACCESS]], i32 0, i32 1
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr %[[MEMBER_ACCESS_2]], ptr %[[OFFLOAD_PTR_ARR]], align 8

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_derived_explicit_multiple_nested_members_{{.*}}
!CHECK: %[[ALLOCA:.*]] = alloca %_QFmaptype_derived_explicit_multiple_nested_membersTscalar_and_array, i64 1, align 8
!CHECK: %[[MEMBER_ACCESS_0:.*]] = getelementptr %_QFmaptype_derived_explicit_multiple_nested_membersTscalar_and_array, ptr %[[ALLOCA]], i32 0, i32 2
!CHECK: %[[MEMBER_ACCESS_1:.*]] = getelementptr %_QFmaptype_derived_explicit_multiple_nested_membersTnested, ptr %[[MEMBER_ACCESS_0]], i32 0, i32 0
!CHECK: %[[MEMBER_ACCESS_2:.*]] = getelementptr %_QFmaptype_derived_explicit_multiple_nested_membersTnested, ptr %[[MEMBER_ACCESS_0]], i32 0, i32 1
!CHECK: %[[ARR_END_OFF:.*]] = getelementptr float, ptr %[[MEMBER_ACCESS_2]], i64 1
!CHECK: %[[ARR_END:.*]] = ptrtoaddr ptr %[[ARR_END_OFF]] to i64
!CHECK: %[[FIRST_MEMBER:.*]] = ptrtoaddr ptr %[[MEMBER_ACCESS_1]] to i64
!CHECK: %[[SIZE:.*]] = sub i64 %[[ARR_END]], %[[FIRST_MEMBER]]
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr %[[MEMBER_ACCESS_1]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_SIZE_ARR:.*]] = getelementptr inbounds [4 x i64], ptr %.offload_sizes, i32 0, i32 0
!CHECK: store i64 %[[SIZE]], ptr %[[OFFLOAD_SIZE_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR_2:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR_2]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR_2:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 1
!CHECK: store ptr %[[MEMBER_ACCESS_1]], ptr %[[OFFLOAD_PTR_ARR_2]], align 8
!CHECK: %[[BASE_PTR_ARR_3:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR_3]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR_3:.*]] = getelementptr inbounds [4 x ptr], ptr %.offload_ptrs, i32 0, i32 2
!CHECK: store ptr %[[MEMBER_ACCESS_2]], ptr %[[OFFLOAD_PTR_ARR_3]], align 8

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_derived_explicit_nested_member_with_bounds_{{.*}}
!CHECK: %[[ALLOCA:.*]] = alloca %_QFmaptype_derived_explicit_nested_member_with_boundsTscalar_and_array, i64 1, align 8
!CHECK: %[[MEMBER_ACCESS:.*]] = getelementptr %_QFmaptype_derived_explicit_nested_member_with_boundsTscalar_and_array, ptr %[[ALLOCA]], i32 0, i32 2
!CHECK: %[[MEMBER_ACCESS_1:.*]] = getelementptr %_QFmaptype_derived_explicit_nested_member_with_boundsTnested, ptr %[[MEMBER_ACCESS]], i32 0, i32 2
!CHECK: %[[ARR_OFF:.*]] = getelementptr inbounds [10 x i32], ptr %[[MEMBER_ACCESS_1]], i64 0, i64 1
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [2 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr %[[ARR_OFF]], ptr %[[OFFLOAD_PTR_ARR]], align 8

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_derived_type_alloca_{{.*}}
!CHECK: %[[ALLOCATABLE_DESC_ALLOCA:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, align 8
!CHECK: %[[ALLOCA:.*]] = alloca %_QFmaptype_derived_type_allocaTone_layer, i64 1, align 8
!CHECK: %[[MEMBER_ACCESS:.*]] = getelementptr %_QFmaptype_derived_type_allocaTone_layer, ptr %[[ALLOCA]], i32 0, i32 4
!CHECK: %[[DESC_BOUND_ACCESS:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[ALLOCATABLE_DESC_ALLOCA]], i32 0, i32 7, i64 0, i32 1
!CHECK: %[[DESC_BOUND_ACCESS_LOAD:.*]] = load i64, ptr %[[DESC_BOUND_ACCESS]], align 8
!CHECK: %[[OFFSET_UB:.*]] = sub i64 %[[DESC_BOUND_ACCESS_LOAD]], 1
!CHECK: %[[MEMBER_DESCRIPTOR_BASE_ADDR:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[MEMBER_ACCESS]], i32 0, i32 0
!CHECK: %[[CALCULATE_DIM_SIZE:.*]] = sub i64 %[[OFFSET_UB]], 0
!CHECK: %[[RESTORE_OFFSET:.*]] = add i64 %[[CALCULATE_DIM_SIZE]], 1
!CHECK: %[[MEMBER_BASE_ADDR_SIZE:.*]] = mul i64 1, %[[RESTORE_OFFSET]]
!CHECK: %[[DESC_BASE_ADDR_DATA_SIZE:.*]] = mul i64 %[[MEMBER_BASE_ADDR_SIZE]], 4
!CHECK: %[[LOAD_ADDR_DATA:.*]] = load ptr, ptr %[[MEMBER_DESCRIPTOR_BASE_ADDR]], align 8
!CHECK: %[[GEP_ADDR_DATA:.*]] = getelementptr inbounds i32, ptr %[[LOAD_ADDR_DATA]], i64 0
!CHECK: %[[LOAD_ADDR_DATA2:.*]] = load ptr, ptr %[[MEMBER_DESCRIPTOR_BASE_ADDR]], align 8
!CHECK: %[[GEP_ADDR_DATA2:.*]] = getelementptr inbounds i32, ptr %[[LOAD_ADDR_DATA2]], i64 0
!CHECK: %[[MEMBER_ACCESS_ADDR_END:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[MEMBER_ACCESS]], i64 1
!CHECK: %[[MEMBER_ACCESS_ADDR_INT:.*]] = ptrtoaddr ptr %[[MEMBER_ACCESS_ADDR_END]] to i64
!CHECK: %[[MEMBER_ACCESS_ADDR_BEGIN:.*]] = ptrtoaddr ptr %[[MEMBER_ACCESS]] to i64
!CHECK: %[[DTYPE_SIZE_CALC:.*]] = sub i64 %[[MEMBER_ACCESS_ADDR_INT]], %[[MEMBER_ACCESS_ADDR_BEGIN]]
!CHECK: %[[DTYPE_CMP:.*]] = icmp eq ptr %[[GEP_ADDR_DATA]], null
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [5 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [5 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr %[[MEMBER_ACCESS]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_SIZE_ARR:.*]] = getelementptr inbounds [5 x i64], ptr %.offload_sizes, i32 0, i32 0
!CHECK: store i64 %[[DTYPE_SIZE_CALC]], ptr %[[OFFLOAD_SIZE_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [5 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [5 x ptr], ptr %.offload_ptrs, i32 0, i32 1
!CHECK: store ptr %[[MEMBER_ACCESS]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [5 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [5 x ptr], ptr %.offload_ptrs, i32 0, i32 2
!CHECK: store ptr %[[GEP_ADDR_DATA2]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [5 x ptr], ptr %.offload_baseptrs, i32 0, i32 3
!CHECK: store ptr %[[MEMBER_ACCESS]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [5 x ptr], ptr %.offload_ptrs, i32 0, i32 3
!CHECK: store ptr %array_offset, ptr %[[OFFLOAD_PTR_ARR]], align 8

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_alloca_derived_type_{{.*}}
!CHECK: %{{.*}} = alloca { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, align 8
!CHECK: %[[DTYPE_DESC_ALLOCA:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, align 8
!CHECK: %[[DTYPE_ARRAY_MEMBER_DESC_ALLOCA:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, align 8
!CHECK: %[[DTYPE_DESC_ALLOCA_2:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, align 8
!CHECK: %[[DTYPE_DESC_ALLOCA_3:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, i64 1, align 8
!CHECK: %{{.*}} = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %{{.*}}, i32 0, i32 0
!CHECK: %{{.*}} = load ptr, ptr %{{.*}}, align 8
!CHECK: %{{.*}} = getelementptr %_QFmaptype_alloca_derived_typeTone_layer, ptr %{{.*}}, i32 0, i32 4
!CHECK: %{{.*}} = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %{{.*}}, i32 0, i32 0
!CHECK: %{{.*}} = load ptr, ptr %{{.*}}, align 8
!CHECK: %{{.*}} = getelementptr %_QFmaptype_alloca_derived_typeTone_layer, ptr %{{.*}}, i32 0, i32 4
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
!CHECK: %[[SIZE_ZERO_CMP:.*]] = icmp eq i64 %[[MEMBER_SIZE_CALC_4]], 0
!CHECK: %[[SIZE_ADJUSTED:.*]] = select i1 %[[SIZE_ZERO_CMP]], i64 1, i64 %[[MEMBER_SIZE_CALC_4]]
!CHECK: %[[DTYPE_BASE_ADDR_LOAD_3:.*]] = load ptr, ptr %[[DTYPE_BASE_ADDR_ACCESS_3]], align 8
!CHECK: %[[DTYPE_BASE_ADDR_LOAD_3_1:.*]] = load ptr, ptr %[[DTYPE_BASE_ADDR_ACCESS_3]], align 8
!CHECK: %[[LOAD_DTYPE_DESC_MEMBER:.*]] = load ptr, ptr %[[DTYPE_ALLOCA_MEMBER_BASE_ADDR_ACCESS]], align 8
!CHECK: %[[MEMBER_ARRAY_OFFSET:.*]] = getelementptr inbounds i32, ptr %[[LOAD_DTYPE_DESC_MEMBER]], i64 0
!CHECK: %[[SIZE_CALC_1:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[DTYPE_DESC_ALLOCA_3]], i32 1
!CHECK: %[[SIZE_CALC_2:.*]] = ptrtoaddr ptr %[[SIZE_CALC_1]] to i64
!CHECK: %[[SIZE_CALC_3:.*]] = ptrtoaddr ptr %[[DTYPE_DESC_ALLOCA_3]] to i64
!CHECK: %[[SIZE_CALC_4:.*]] = sub i64 %[[SIZE_CALC_2]], %[[SIZE_CALC_3]]
!CHECK: %[[NULL_CMP:.*]] = icmp eq ptr %[[DTYPE_BASE_ADDR_LOAD_3_1]], null
!CHECK: %[[SEL_SZ:.*]] = select i1 %[[NULL_CMP]], i64 0, i64 136
!CHECK: %[[NULL_CMP2:.*]] = icmp eq ptr %{{.*}}, null
!CHECK: %[[SEL_SZ2:.*]] = select i1 %[[NULL_CMP2]], i64 0, i64 %[[SIZE_ADJUSTED]]
!CHECK: getelementptr inbounds [9 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: getelementptr inbounds [9 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: getelementptr inbounds [9 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
!CHECK: getelementptr inbounds [9 x ptr], ptr %.offload_ptrs, i32 0, i32 1
!CHECK: getelementptr inbounds [9 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
!CHECK: getelementptr inbounds [9 x ptr], ptr %.offload_ptrs, i32 0, i32 2
!CHECK: getelementptr inbounds [9 x ptr], ptr %.offload_baseptrs, i32 0, i32 3
!CHECK: getelementptr inbounds [9 x ptr], ptr %.offload_ptrs, i32 0, i32 3
!CHECK: getelementptr inbounds [9 x ptr], ptr %.offload_baseptrs, i32 0, i32 4
!CHECK: getelementptr inbounds [9 x ptr], ptr %.offload_ptrs, i32 0, i32 4
!CHECK: getelementptr inbounds [9 x ptr], ptr %.offload_baseptrs, i32 0, i32 5
!CHECK: getelementptr inbounds [9 x ptr], ptr %.offload_ptrs, i32 0, i32 5

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
!CHECK: %[[SIZE_ZERO_CMP:.*]] = icmp eq i64 %[[ALLOCATABLE_MEMBER_SIZE_CALC_5]], 0
!CHECK: %[[SIZE_ADJUSTED:.*]] = select i1 %[[SIZE_ZERO_CMP]], i64 1, i64 %[[ALLOCATABLE_MEMBER_SIZE_CALC_5]]
!CHECK: %[[LOAD_BASE_ADDR:.*]] = load ptr, ptr %[[DTYPE_DESC_BASE_ADDR]], align 8
!CHECK: %[[LOAD_BASE_ADDR2:.*]] = load ptr, ptr %[[DTYPE_DESC_BASE_ADDR]], align 8
!CHECK: %[[LOAD_DESC_MEMBER_BASE_ADDR:.*]] = load ptr, ptr %[[MAPPED_MEMBER_BASE_ADDR_ACCESS]], align 8
!CHECK: %[[ARRAY_OFFSET:.*]] = getelementptr inbounds i32, ptr %[[LOAD_DESC_MEMBER_BASE_ADDR]], i64 0
!CHECK: %[[NULL_CMP:.*]] = icmp eq ptr %[[LOAD_BASE_ADDR2]], null
!CHECK: %[[SEL_SZ:.*]] = select i1 %[[NULL_CMP]], i64 0, i64 240
!CHECK: %[[NULL_CMP2:.*]] = icmp eq ptr %[[ARRAY_OFFSET]], null
!CHECK: %[[SEL_SZ2:.*]] = select i1 %[[NULL_CMP2]], i64 0, i64 %[[SIZE_ADJUSTED]]
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: store ptr %[[DTYPE_DESC_ALLOCA_3]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr %[[DTYPE_DESC_ALLOCA_3]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [9 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
!CHECK: store ptr %[[DTYPE_DESC_ALLOCA_3]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: getelementptr inbounds [9 x ptr], ptr %.offload_ptrs, i32 0, i32 1
!CHECK: getelementptr inbounds [9 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
!CHECK: getelementptr inbounds [9 x ptr], ptr %.offload_ptrs, i32 0, i32 2
!CHECK: getelementptr inbounds [9 x ptr], ptr %.offload_baseptrs, i32 0, i32 3
!CHECK: getelementptr inbounds [9 x ptr], ptr %.offload_ptrs, i32 0, i32 3
!CHECK: getelementptr inbounds [9 x ptr], ptr %.offload_baseptrs, i32 0, i32 4
!CHECK: getelementptr inbounds [9 x ptr], ptr %.offload_ptrs, i32 0, i32 4
!CHECK: getelementptr inbounds [9 x ptr], ptr %.offload_baseptrs, i32 0, i32 5
!CHECK: getelementptr inbounds [9 x ptr], ptr %.offload_ptrs, i32 0, i32 5

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_nested_derived_type_alloca{{.*}}
!CHECK: %[[ALLOCATABLE_MEMBER_ALLOCA:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, align 8
!CHECK: %[[ALLOCA:.*]] = alloca %_QFmaptype_nested_derived_type_allocaTtop_layer, i64 1, align 8
!CHECK: %[[NESTED_DTYPE_MEMBER_ACCESS:.*]] = getelementptr %_QFmaptype_nested_derived_type_allocaTtop_layer, ptr %[[ALLOCA]], i32 0, i32 6
!CHECK: %[[NESTED_MEMBER_ACCESS:.*]] = getelementptr %_QFmaptype_nested_derived_type_allocaTmiddle_layer, ptr %[[NESTED_DTYPE_MEMBER_ACCESS]], i32 0, i32 2
!CHECK: %[[ALLOCATABLE_MEMBER_BASE_ADDR:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[ALLOCATABLE_MEMBER_ALLOCA]], i32 0, i32 7, i64 0, i32 1
!CHECK: %[[ALLOCATABLE_MEMBER_ADDR_LOAD:.*]] = load i64, ptr %[[ALLOCATABLE_MEMBER_BASE_ADDR]], align 8
!CHECK: %[[ALLOCATABLE_MEMBER_SIZE_CALC_1:.*]] = sub i64 %[[ALLOCATABLE_MEMBER_ADDR_LOAD]], 1
!CHECK: %[[NESTED_MEMBER_BASE_ADDR_ACCESS:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %{{.*}}, i32 0, i32 0
!CHECK: %[[ALLOCATABLE_MEMBER_SIZE_CALC_2:.*]] = sub i64 %[[ALLOCATABLE_MEMBER_SIZE_CALC_1]], 0
!CHECK: %[[ALLOCATABLE_MEMBER_SIZE_CALC_3:.*]] = add i64 %[[ALLOCATABLE_MEMBER_SIZE_CALC_2]], 1
!CHECK: %[[ALLOCATABLE_MEMBER_SIZE_CALC_4:.*]] = mul i64 1, %[[ALLOCATABLE_MEMBER_SIZE_CALC_3]]
!CHECK: %[[ALLOCATABLE_MEMBER_SIZE_CALC_5:.*]] = mul i64 %[[ALLOCATABLE_MEMBER_SIZE_CALC_4]], 4
!CHECK: %[[SIZE_ZERO_CMP:.*]] = icmp eq i64 %[[ALLOCATABLE_MEMBER_SIZE_CALC_5]], 0
!CHECK: %[[SIZE_ADJUSTED:.*]] = select i1 %[[SIZE_ZERO_CMP]], i64 1, i64 %[[ALLOCATABLE_MEMBER_SIZE_CALC_5]]
!CHECK: %[[LOAD_BASE_ADDR:.*]] = load ptr, ptr %[[NESTED_MEMBER_BASE_ADDR_ACCESS]], align 8
!CHECK: %[[ARR_OFFS:.*]] = getelementptr inbounds i32, ptr %[[LOAD_BASE_ADDR]], i64 0
!CHECK: %[[LOAD_BASE_ADDR:.*]] = load ptr, ptr %[[NESTED_MEMBER_BASE_ADDR_ACCESS]], align 8
!CHECK: %[[ARR_OFFS2:.*]] = getelementptr inbounds i32, ptr %[[LOAD_BASE_ADDR]], i64 0
!CHECK: %[[NESTED_MEMBER_BASE_ADDR_ACCESS_2:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[NESTED_MEMBER_ACCESS]], i64 1
!CHECK: %[[DTYPE_SEGMENT_SIZE_CALC_1:.*]] = ptrtoaddr ptr %[[NESTED_MEMBER_BASE_ADDR_ACCESS_2]] to i64
!CHECK: %[[DTYPE_SEGMENT_SIZE_CALC_2:.*]] = ptrtoaddr ptr %[[NESTED_MEMBER_ACCESS]] to i64
!CHECK: %[[DTYPE_SEGMENT_SIZE_CALC_3:.*]] = sub i64 %[[DTYPE_SEGMENT_SIZE_CALC_1]], %[[DTYPE_SEGMENT_SIZE_CALC_2]]
!CHECK: %[[DATA_CMP:.*]] = icmp eq ptr %[[ARR_OFFS2]], null
!CHECK: %[[DATA_SEL:.*]] = select i1 %[[DATA_CMP]], i64 0, i64 %[[SIZE_ADJUSTED]]
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [5 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [5 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr %[[NESTED_MEMBER_ACCESS]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_SIZE_ARR:.*]] = getelementptr inbounds [5 x i64], ptr %.offload_sizes, i32 0, i32 0
!CHECK: store i64 %[[DTYPE_SEGMENT_SIZE_CALC_3]], ptr %[[OFFLOAD_SIZE_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [5 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [5 x ptr], ptr %.offload_ptrs, i32 0, i32 1
!CHECK: store ptr %[[NESTED_MEMBER_ACCESS]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [5 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
!CHECK: store ptr %[[ALLOCA]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [5 x ptr], ptr %.offload_ptrs, i32 0, i32 2
!CHECK: store ptr %[[ARR_OFFS2]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_SIZE_ARR:.*]] = getelementptr inbounds [5 x i64], ptr %.offload_sizes, i32 0, i32 2
!CHECK: store i64 %[[DATA_SEL]], ptr %[[OFFLOAD_SIZE_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [5 x ptr], ptr %.offload_baseptrs, i32 0, i32 3
!CHECK: store ptr %[[NESTED_MEMBER_ACCESS]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [5 x ptr], ptr %.offload_ptrs, i32 0, i32 3
!CHECK: store ptr %[[ARR_OFFS]], ptr %[[OFFLOAD_PTR_ARR]], align 8

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_nested_derived_type_member_idx{{.*}}
!CHECK: %[[ALLOCA_0:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, align 8
!CHECK: %[[ALLOCA_1:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, align 8
!CHECK: %[[ALLOCA:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, align 8
!CHECK: %[[BASE_PTR_1:.*]] = alloca %_QFmaptype_nested_derived_type_member_idxTdtype, i64 1, align 8
!CHECK: %[[OFF_PTR_1:.*]] = getelementptr %_QFmaptype_nested_derived_type_member_idxTdtype, ptr %[[BASE_PTR_1]], i32 0, i32 1
!CHECK: %[[BOUNDS_ACC:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %[[ALLOCA]], i32 0, i32 7, i64 0, i32 1
!CHECK: %[[BOUNDS_LD:.*]] = load i64, ptr %[[BOUNDS_ACC]], align 8
!CHECK: %[[BOUNDS_ACC_2:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[ALLOCA_1]], i32 0, i32 7, i64 0, i32 1
!CHECK: %[[BOUNDS_LD_2:.*]] = load i64, ptr %[[BOUNDS_ACC_2]], align 8
!CHECK: %[[BOUNDS_CALC:.*]] = sub i64 %[[BOUNDS_LD_2]], 1
!CHECK: %[[OFF_PTR_CALC_0:.*]] = sub i64 %[[BOUNDS_LD]], 1
!CHECK: %[[OFF_PTR_2:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %[[OFF_PTR_1]], i32 0, i32 0
!CHECK: %[[GEP_LB:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %[[ALLOCA_0]], i32 0, i32 7, i64 0, i32 0
!CHECK: %[[LOAD_LB:.*]] = load i64, ptr %[[GEP_LB]], align 8
!CHECK: %[[GEP_UB:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %[[ALLOCA_0]], i32 0, i32 7, i64 0, i32 1
!CHECK: %[[LOAD_UB:.*]] = load i64, ptr %[[GEP_UB]], align 8
!CHECK: %[[GEP_DESC_PTR:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %[[ALLOCA_0]], i32 0, i32 0
!CHECK: %[[SZ_CALC_1:.*]] = load ptr, ptr %[[GEP_DESC_PTR]], align 8
!CHECK: %[[SZ_CALC_2:.*]] = sub nsw i64 2, %[[LOAD_LB]]
!CHECK: %[[SZ_CALC_3:.*]] = mul nuw nsw i64 %[[SZ_CALC_2]], 1
!CHECK: %[[SZ_CALC_4:.*]] = mul nuw nsw i64 %[[SZ_CALC_3]], 1
!CHECK: %[[SZ_CALC_5:.*]] = add nuw nsw i64 %[[SZ_CALC_4]], 0
!CHECK: %[[SZ_CALC_6:.*]] = mul nuw nsw i64 1, %[[LOAD_UB]]
!CHECK: %[[SZ_CALC_7:.*]] = getelementptr nusw nuw %_QFmaptype_nested_derived_type_member_idxTvertexes, ptr %[[SZ_CALC_1]], i64 %[[SZ_CALC_5]]
!CHECK: %[[SZ_CALC_8:.*]] = getelementptr %_QFmaptype_nested_derived_type_member_idxTvertexes, ptr %[[SZ_CALC_7]], i32 0, i32 2
!CHECK: %[[OFF_PTR_4:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[SZ_CALC_8]], i32 0, i32 0
!CHECK: %[[OFF_PTR_CALC_1:.*]] = sub i64 %[[OFF_PTR_CALC_0]], 0
!CHECK: %[[OFF_PTR_CALC_2:.*]] = add i64 %[[OFF_PTR_CALC_1]], 1
!CHECK: %[[OFF_PTR_CALC_3:.*]] = mul i64 1, %[[OFF_PTR_CALC_2]]
!CHECK: %[[OFF_PTR_3:.*]] = mul i64 %[[OFF_PTR_CALC_3]], 104
!CHECK: %[[SIZE_ZERO_CMP_1:.*]] = icmp eq i64 %[[OFF_PTR_3]], 0
!CHECK: %[[SIZE_ADJUSTED_1:.*]] = select i1 %[[SIZE_ZERO_CMP_1]], i64 1, i64 %[[OFF_PTR_3]]
!CHECK: %[[SZ_CALC_1_2:.*]] = sub i64 %[[BOUNDS_CALC]], 0
!CHECK: %[[SZ_CALC_2_2:.*]] = add i64 %[[SZ_CALC_1_2]], 1
!CHECK: %[[SZ_CALC_3_2:.*]] = mul i64 1, %[[SZ_CALC_2_2]]
!CHECK: %[[SZ_CALC_4_2:.*]] = mul i64 %[[SZ_CALC_3_2]], 4
!CHECK: %[[SIZE_ZERO_CMP_2:.*]] = icmp eq i64 %[[SZ_CALC_4_2]], 0
!CHECK: %[[SIZE_ADJUSTED_2:.*]] = select i1 %[[SIZE_ZERO_CMP_2]], i64 1, i64 %[[SZ_CALC_4_2]]
!CHECK: %[[LOAD_OFF_PTR:.*]] = load ptr, ptr %[[OFF_PTR_2]], align 8
!CHECK: %[[ARR_OFFS:.*]] = getelementptr inbounds %_QFmaptype_nested_derived_type_member_idxTvertexes, ptr %[[LOAD_OFF_PTR]], i64 0
!CHECK: %[[LOAD_ARR_OFFS:.*]] = load ptr, ptr %[[OFF_PTR_4]], align 8
!CHECK: %[[ARR_OFFS_1:.*]] = getelementptr inbounds i32, ptr %[[LOAD_ARR_OFFS]], i64 0
!CHECK: %[[LOAD_OFF_PTR:.*]] = load ptr, ptr %[[OFF_PTR_2]], align 8
!CHECK: %[[ARR_OFFS_2:.*]] = getelementptr inbounds %_QFmaptype_nested_derived_type_member_idxTvertexes, ptr %[[LOAD_OFF_PTR]], i64 0
!CHECK: %[[LOAD_ARR_OFFS:.*]] = load ptr, ptr %[[OFF_PTR_4]], align 8
!CHECK: %[[ARR_OFFS_3:.*]] = getelementptr inbounds i32, ptr %[[LOAD_ARR_OFFS]], i64 0
!CHECK: %[[SZ_CALC_1:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %[[OFF_PTR_1]], i64 1
!CHECK: %[[SZ_CALC_2:.*]] = ptrtoaddr ptr %[[SZ_CALC_1]] to i64
!CHECK: %[[SZ_CALC_3:.*]] = ptrtoaddr ptr %[[OFF_PTR_1]] to i64
!CHECK: %[[SZ_CALC_4:.*]] = sub i64 %[[SZ_CALC_2]], %[[SZ_CALC_3]]
!CHECK: %[[SIZE_CMP:.*]] = icmp eq ptr %[[ARR_OFFS_2]], null
!CHECK: %[[SIZE_SEL:.*]] = select i1 %[[SIZE_CMP]], i64 0, i64 %[[SIZE_ADJUSTED_1]]
!CHECK: %[[SIZE_CMP2:.*]] = icmp eq ptr %[[ARR_OFFS_3]], null
!CHECK: %[[SIZE_SEL2:.*]] = select i1 %[[SIZE_CMP2]], i64 0, i64 %[[SIZE_ADJUSTED_2]]
!CHECK: %[[SIZE_CMP3:.*]] = icmp eq ptr %[[ARR_OFFS]], null
!CHECK: %[[SIZE_SEL3:.*]] = select i1 %[[SIZE_CMP3]], i64 0, i64 64
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: store ptr %[[BASE_PTR_1]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr %[[OFF_PTR_1]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_SIZE_ARR:.*]] = getelementptr inbounds [8 x i64], ptr %.offload_sizes, i32 0, i32 0
!CHECK: store i64 %[[SZ_CALC_4]], ptr %[[OFFLOAD_SIZE_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
!CHECK: store ptr %[[BASE_PTR_1]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_ptrs, i32 0, i32 1
!CHECK: store ptr %[[OFF_PTR_1]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
!CHECK: store ptr %[[BASE_PTR_1]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_ptrs, i32 0, i32 2
!CHECK: store ptr %[[ARR_OFFS_2]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_SIZE_ARR:.*]] = getelementptr inbounds [8 x i64], ptr %.offload_sizes, i32 0, i32 2
!CHECK: store i64 %[[SIZE_SEL]], ptr %[[OFFLOAD_SIZE_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_baseptrs, i32 0, i32 3
!CHECK: store ptr %[[BASE_PTR_1]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_ptrs, i32 0, i32 3
!CHECK: store ptr %[[SZ_CALC_8]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_baseptrs, i32 0, i32 4
!CHECK: store ptr %[[BASE_PTR_1]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_ptrs, i32 0, i32 4
!CHECK: store ptr %[[ARR_OFFS_3]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_baseptrs, i32 0, i32 5
!CHECK: store ptr %[[OFF_PTR_1]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_ptrs, i32 0, i32 5
!CHECK: store ptr %[[ARR_OFFS]], ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_SIZE_ARR:.*]] = getelementptr inbounds [8 x i64], ptr %.offload_sizes, i32 0, i32 5
!CHECK: store i64 %[[SIZE_SEL3]], ptr %[[OFFLOAD_SIZE_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_baseptrs, i32 0, i32 6
!CHECK: store ptr %[[SZ_CALC_8]], ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [8 x ptr], ptr %.offload_ptrs, i32 0, i32 6
!CHECK: store ptr %[[ARR_OFFS_1]], ptr %[[OFFLOAD_PTR_ARR]], align 8

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_common_block_members_{{.*}}
!CHECK: %[[BASE_PTR_ARR:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: store ptr @var_common_, ptr %[[BASE_PTR_ARR]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr @var_common_, ptr %[[OFFLOAD_PTR_ARR]], align 8
!CHECK: %[[BASE_PTR_ARR_1:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
!CHECK: store ptr getelementptr inbounds nuw (i8, ptr @var_common_, i64 4), ptr %[[BASE_PTR_ARR_1]], align 8
!CHECK: %[[OFFLOAD_PTR_ARR_1:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_ptrs, i32 0, i32 1
!CHECK: store ptr getelementptr inbounds nuw (i8, ptr @var_common_, i64 4), ptr %[[OFFLOAD_PTR_ARR_1]], align 8

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_polymorphic_allocatable_explicit_{{.*}}
!CHECK: %.offload_baseptrs = alloca [7 x ptr], align 8
!CHECK: %.offload_ptrs = alloca [7 x ptr], align 8
!CHECK: %.offload_mappers = alloca [7 x ptr], align 8
!CHECK: %.offload_sizes = alloca [7 x i64], align 8
!CHECK: %[[BASE_ADDR_FIELD:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[DESC:.*]], i32 0, i32 0
!CHECK: %[[ELEM_LEN_FIELD:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %{{.*}}, i32 0, i32 1
!CHECK: %[[ELEM_LEN:.*]] = load i64, ptr %[[ELEM_LEN_FIELD]], align 8
!CHECK: %[[UB:.*]] = sub i64 %[[ELEM_LEN]], 1
!CHECK: %[[RTTI_FIELD:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[DESC]], i32 0, i32 7
!CHECK: %[[COUNT_SUB:.*]] = sub i64 %[[UB]], 0
!CHECK: %[[COUNT_ADD:.*]] = add i64 %[[COUNT_SUB]], 1
!CHECK: %[[COUNT_MUL:.*]] = mul i64 1, %[[COUNT_ADD]]
!CHECK: %[[ELEMENT_COUNT:.*]] = mul i64 %[[COUNT_MUL]], 1
!CHECK: %[[COUNT_CMP:.*]] = icmp eq i64 %[[ELEMENT_COUNT]], 0
!CHECK: %[[COUNT_SEL:.*]] = select i1 %[[COUNT_CMP]], i64 1, i64 %[[ELEMENT_COUNT]]
!CHECK: %[[LOAD_BASE_ADDR:.*]] = load ptr, ptr %[[BASE_ADDR_FIELD]], align 8
!CHECK: %[[ARRAY_OFFSET:.*]] = getelementptr inbounds i8, ptr %[[LOAD_BASE_ADDR]], i64 0
!CHECK: %[[LOAD_RTTI:.*]] = load ptr, ptr %[[RTTI_FIELD]], align 8
!CHECK: %[[LOAD_BASE_ADDR2:.*]] = load ptr, ptr %[[BASE_ADDR_FIELD]], align 8
!CHECK: %[[ARRAY_OFFSET1:.*]] = getelementptr inbounds i8, ptr %[[LOAD_BASE_ADDR2]], i64 0
!CHECK: %[[DESC_END:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[DESC]], i32 1
!CHECK: %[[DESC_END_INT:.*]] = ptrtoaddr ptr %[[DESC_END]] to i64
!CHECK: %[[DESC_BEGIN_INT:.*]] = ptrtoaddr ptr %[[DESC]] to i64
!CHECK: %[[DESC_SIZE:.*]] = sub i64 %[[DESC_END_INT]], %[[DESC_BEGIN_INT]]
!CHECK: %[[DATA_CMP:.*]] = icmp eq ptr %[[ARRAY_OFFSET1]], null
!CHECK: %[[DATA_SIZE:.*]] = select i1 %[[DATA_CMP]], i64 0, i64 %[[COUNT_SEL]]
!CHECK: %[[SEG_CMP:.*]] = icmp eq ptr %[[ARRAY_OFFSET]], null
!CHECK: %[[SEG_SIZE:.*]] = select i1 %[[SEG_CMP]], i64 0, i64 40
!CHECK: %[[ATTACH_CMP:.*]] = icmp eq ptr %[[LOAD_RTTI]], null
!CHECK: %[[ATTACH_SIZE:.*]] = select i1 %[[ATTACH_CMP]], i64 0, i64 8
!CHECK: call void @llvm.memcpy{{.*}}ptr align 8 %.offload_sizes, ptr align 8 @.offload_sizes{{.*}}
!CHECK: %[[BASE0:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: store ptr %[[DESC]], ptr %[[BASE0]], align 8
!CHECK: %[[PTR0:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr %[[DESC]], ptr %[[PTR0]], align 8
!CHECK: %[[SIZE0:.*]] = getelementptr inbounds [7 x i64], ptr %.offload_sizes, i32 0, i32 0
!CHECK: store i64 %[[DESC_SIZE]], ptr %[[SIZE0]], align 8
!CHECK: %[[BASE1:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
!CHECK: store ptr %[[DESC]], ptr %[[BASE1]], align 8
!CHECK: %[[PTR1:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 1
!CHECK: store ptr %[[DESC]], ptr %[[PTR1]], align 8
!CHECK: %[[BASE2:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
!CHECK: store ptr %[[DESC]], ptr %[[BASE2]], align 8
!CHECK: %[[PTR2:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 2
!CHECK: store ptr %[[ARRAY_OFFSET1]], ptr %[[PTR2]], align 8
!CHECK: %[[SIZE2:.*]] = getelementptr inbounds [7 x i64], ptr %.offload_sizes, i32 0, i32 2
!CHECK: store i64 %[[DATA_SIZE]], ptr %[[SIZE2]], align 8
!CHECK: %[[BASE3:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 3
!CHECK: store ptr %[[RES:.*]], ptr %[[BASE3]], align 8
!CHECK: %[[PTR3:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 3
!CHECK: store ptr %[[RES]], ptr %[[PTR3]], align 8
!CHECK: %[[BASE4:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 4
!CHECK: store ptr %[[DESC]], ptr %[[BASE4]], align 8
!CHECK: %[[PTR4:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 4
!CHECK: store ptr %[[ARRAY_OFFSET]], ptr %[[PTR4]], align 8
!CHECK: %[[SIZE4:.*]] = getelementptr inbounds [7 x i64], ptr %.offload_sizes, i32 0, i32 4
!CHECK: store i64 %[[SEG_SIZE]], ptr %[[SIZE4]], align 8
!CHECK: %[[BASE5:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 5
!CHECK: store ptr %[[RTTI_FIELD]], ptr %[[BASE5]], align 8
!CHECK: %[[PTR5:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 5
!CHECK: store ptr %[[LOAD_RTTI]], ptr %[[PTR5]], align 8
!CHECK: %[[SIZE5:.*]] = getelementptr inbounds [7 x i64], ptr %.offload_sizes, i32 0, i32 5
!CHECK: store i64 %[[ATTACH_SIZE]], ptr %[[SIZE5]], align 8
!CHECK: %[[BASE6:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 6
!CHECK: store ptr null, ptr %[[BASE6]], align 8
!CHECK: %[[PTR6:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 6
!CHECK: store ptr null, ptr %[[PTR6]], align 8

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_polymorphic_allocatable_implicit_{{.*}}
!CHECK: %.offload_baseptrs = alloca [7 x ptr], align 8
!CHECK: %.offload_ptrs = alloca [7 x ptr], align 8
!CHECK: %.offload_mappers = alloca [7 x ptr], align 8
!CHECK: %.offload_sizes = alloca [7 x i64], align 8
!CHECK: %[[BASE_ADDR_FIELD:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[DESC:.*]], i32 0, i32 0
!CHECK: %[[ELEM_LEN_FIELD:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %{{.*}}, i32 0, i32 1
!CHECK: %[[ELEM_LEN:.*]] = load i64, ptr %[[ELEM_LEN_FIELD]], align 8
!CHECK: %[[UB:.*]] = sub i64 %[[ELEM_LEN]], 1
!CHECK: %[[RTTI_FIELD:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[DESC]], i32 0, i32 7
!CHECK: %[[COUNT_SEL:.*]] = select i1 %{{.*}}, i64 1, i64 %{{.*}}
!CHECK: %[[LOAD_BASE_ADDR:.*]] = load ptr, ptr %[[BASE_ADDR_FIELD]], align 8
!CHECK: %[[ARRAY_OFFSET:.*]] = getelementptr inbounds i8, ptr %[[LOAD_BASE_ADDR]], i64 0
!CHECK: %[[LOAD_RTTI:.*]] = load ptr, ptr %[[RTTI_FIELD]], align 8
!CHECK: %[[LOAD_BASE_ADDR2:.*]] = load ptr, ptr %[[BASE_ADDR_FIELD]], align 8
!CHECK: %[[ARRAY_OFFSET1:.*]] = getelementptr inbounds i8, ptr %[[LOAD_BASE_ADDR2]], i64 0
!CHECK: %[[DESC_END:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[DESC]], i32 1
!CHECK: %[[DESC_END_INT:.*]] = ptrtoaddr ptr %[[DESC_END]] to i64
!CHECK: %[[DESC_BEGIN_INT:.*]] = ptrtoaddr ptr %[[DESC]] to i64
!CHECK: %[[DESC_SIZE:.*]] = sub i64 %[[DESC_END_INT]], %[[DESC_BEGIN_INT]]
!CHECK: %[[DATA_CMP:.*]] = icmp eq ptr %[[ARRAY_OFFSET1]], null
!CHECK: %[[DATA_SIZE:.*]] = select i1 %[[DATA_CMP]], i64 0, i64 %[[COUNT_SEL]]
!CHECK: %[[SEG_CMP:.*]] = icmp eq ptr %[[ARRAY_OFFSET]], null
!CHECK: %[[SEG_SIZE:.*]] = select i1 %[[SEG_CMP]], i64 0, i64 40
!CHECK: %[[ATTACH_CMP:.*]] = icmp eq ptr %[[LOAD_RTTI]], null
!CHECK: %[[ATTACH_SIZE:.*]] = select i1 %[[ATTACH_CMP]], i64 0, i64 8
!CHECK: call void @llvm.memcpy{{.*}}ptr align 8 %.offload_sizes, ptr align 8 @.offload_sizes{{.*}}
!CHECK: %[[BASE0:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: store ptr %[[RES:.*]], ptr %[[BASE0]], align 8
!CHECK: %[[PTR0:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr %[[RES]], ptr %[[PTR0]], align 8
!CHECK: %[[BASE1:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
!CHECK: store ptr %[[DESC]], ptr %[[BASE1]], align 8
!CHECK: %[[PTR1:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 1
!CHECK: store ptr %[[DESC]], ptr %[[PTR1]], align 8
!CHECK: %[[SIZE1:.*]] = getelementptr inbounds [7 x i64], ptr %.offload_sizes, i32 0, i32 1
!CHECK: store i64 %[[DESC_SIZE]], ptr %[[SIZE1]], align 8
!CHECK: %[[BASE2:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
!CHECK: store ptr %[[DESC]], ptr %[[BASE2]], align 8
!CHECK: %[[PTR2:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 2
!CHECK: store ptr %[[DESC]], ptr %[[PTR2]], align 8
!CHECK: %[[BASE3:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 3
!CHECK: store ptr %[[DESC]], ptr %[[BASE3]], align 8
!CHECK: %[[PTR3:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 3
!CHECK: store ptr %[[ARRAY_OFFSET1]], ptr %[[PTR3]], align 8
!CHECK: %[[SIZE3:.*]] = getelementptr inbounds [7 x i64], ptr %.offload_sizes, i32 0, i32 3
!CHECK: store i64 %[[DATA_SIZE]], ptr %[[SIZE3]], align 8
!CHECK: %[[BASE4:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 4
!CHECK: store ptr %[[DESC]], ptr %[[BASE4]], align 8
!CHECK: %[[PTR4:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 4
!CHECK: store ptr %[[ARRAY_OFFSET]], ptr %[[PTR4]], align 8
!CHECK: %[[SIZE4:.*]] = getelementptr inbounds [7 x i64], ptr %.offload_sizes, i32 0, i32 4
!CHECK: store i64 %[[SEG_SIZE]], ptr %[[SIZE4]], align 8
!CHECK: %[[BASE5:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 5
!CHECK: store ptr %[[RTTI_FIELD]], ptr %[[BASE5]], align 8
!CHECK: %[[PTR5:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 5
!CHECK: store ptr %[[LOAD_RTTI]], ptr %[[PTR5]], align 8
!CHECK: %[[SIZE5:.*]] = getelementptr inbounds [7 x i64], ptr %.offload_sizes, i32 0, i32 5
!CHECK: store i64 %[[ATTACH_SIZE]], ptr %[[SIZE5]], align 8
!CHECK: %[[BASE6:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 6
!CHECK: store ptr null, ptr %[[BASE6]], align 8
!CHECK: %[[PTR6:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 6
!CHECK: store ptr null, ptr %[[PTR6]], align 8

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_polymorphic_nested_implicit_{{.*}}
!CHECK: %.offload_baseptrs = alloca [3 x ptr], align 8
!CHECK: %.offload_ptrs = alloca [3 x ptr], align 8
!CHECK: %.offload_mappers = alloca [3 x ptr], align 8
!CHECK: %[[BASE0:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: store ptr %[[RES:.*]], ptr %[[BASE0]], align 8
!CHECK: %[[PTR0:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr %[[RES]], ptr %[[PTR0]], align 8
!CHECK: %[[MAPPER0:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_mappers, i64 0, i64 0
!CHECK: store ptr null, ptr %[[MAPPER0]], align 8
!CHECK: %[[BASE1:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
!CHECK: store ptr %[[WRAPPER:.*]], ptr %[[BASE1]], align 8
!CHECK: %[[PTR1:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_ptrs, i32 0, i32 1
!CHECK: store ptr %[[WRAPPER]], ptr %[[PTR1]], align 8
!CHECK: %[[MAPPER1:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_mappers, i64 0, i64 1
!CHECK: store ptr @.omp_mapper.{{.*}}wrapper_t_omp_default_mapper, ptr %[[MAPPER1]], align 8
!CHECK: %[[BASE2:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
!CHECK: store ptr null, ptr %[[BASE2]], align 8
!CHECK: %[[PTR2:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_ptrs, i32 0, i32 2
!CHECK: store ptr null, ptr %[[PTR2]], align 8
!CHECK: %[[MAPPER2:.*]] = getelementptr inbounds [3 x ptr], ptr %.offload_mappers, i64 0, i64 2
!CHECK: store ptr null, ptr %[[MAPPER2]], align 8

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_polymorphic_array_section_explicit_{{.*}}
!CHECK: %.offload_baseptrs = alloca [7 x ptr], align 8
!CHECK: %.offload_ptrs = alloca [7 x ptr], align 8
!CHECK: %.offload_mappers = alloca [7 x ptr], align 8
!CHECK: %.offload_sizes = alloca [7 x i64], align 8
!CHECK: %[[SECTION_LB_ACCESS:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %{{.*}}, i32 0, i32 7, i64 0, i32 0
!CHECK: %[[SECTION_LB:.*]] = load i64, ptr %[[SECTION_LB_ACCESS]], align 8
!CHECK: %[[LB_OFFSET:.*]] = sub i64 2, %[[SECTION_LB]]
!CHECK: %[[UB_OFFSET:.*]] = sub i64 5, %[[SECTION_LB]]
!CHECK: %[[BASE_ADDR_FIELD:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %[[DESC:.*]], i32 0, i32 0
!CHECK: %[[ELEM_LEN_FIELD:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %{{.*}}, i32 0, i32 1
!CHECK: %[[ELEM_LEN:.*]] = load i64, ptr %[[ELEM_LEN_FIELD]], align 8
!CHECK: %[[EXTENT:.*]] = sub i64 %[[UB_OFFSET]], %[[LB_OFFSET]]
!CHECK: %[[EXTENT_1:.*]] = add i64 %[[EXTENT]], 1
!CHECK: %[[LB_BYTE_OFFSET:.*]] = mul i64 %[[LB_OFFSET]], %[[ELEM_LEN]]
!CHECK: %[[EXTENT_BYTES:.*]] = mul i64 %[[EXTENT_1]], %[[ELEM_LEN]]
!CHECK: %[[UB_BYTES:.*]] = add i64 %[[LB_BYTE_OFFSET]], %[[EXTENT_BYTES]]
!CHECK: %[[UB_BYTES_1:.*]] = sub i64 %[[UB_BYTES]], 1
!CHECK: %[[ELEM_LEN_FIELD2:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %{{.*}}, i32 0, i32 1
!CHECK: %[[ELEM_LEN2:.*]] = load i64, ptr %[[ELEM_LEN_FIELD2]], align 8
!CHECK: %[[SEG_LB_BYTES:.*]] = mul i64 %[[LB_OFFSET]], %[[ELEM_LEN2]]
!CHECK: %[[RTTI_FIELD:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %[[DESC]], i32 0, i32 8
!CHECK: %[[COUNT_RANGE:.*]] = sub i64 %[[UB_BYTES_1]], %[[LB_BYTE_OFFSET]]
!CHECK: %[[COUNT_RANGE_1:.*]] = add i64 %[[COUNT_RANGE]], 1
!CHECK: %[[COUNT_MUL:.*]] = mul i64 1, %[[COUNT_RANGE_1]]
!CHECK: %[[ELEMENT_COUNT:.*]] = mul i64 %[[COUNT_MUL]], 1
!CHECK: %[[COUNT_CMP:.*]] = icmp eq i64 %[[ELEMENT_COUNT]], 0
!CHECK: %[[COUNT_SEL:.*]] = select i1 %[[COUNT_CMP]], i64 1, i64 %[[ELEMENT_COUNT]]
!CHECK: %[[LOAD_BASE_ADDR:.*]] = load ptr, ptr %[[BASE_ADDR_FIELD]], align 8
!CHECK: %[[ARRAY_OFFSET:.*]] = getelementptr inbounds i8, ptr %[[LOAD_BASE_ADDR]], i64 %[[SEG_LB_BYTES]]
!CHECK: %[[LOAD_RTTI:.*]] = load ptr, ptr %[[RTTI_FIELD]], align 8
!CHECK: %[[LOAD_BASE_ADDR2:.*]] = load ptr, ptr %[[BASE_ADDR_FIELD]], align 8
!CHECK: %[[ARRAY_OFFSET1:.*]] = getelementptr inbounds i8, ptr %[[LOAD_BASE_ADDR2]], i64 %[[LB_BYTE_OFFSET]]
!CHECK: %[[DESC_END:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %[[DESC]], i32 1
!CHECK: %[[DESC_END_INT:.*]] = ptrtoaddr ptr %[[DESC_END]] to i64
!CHECK: %[[DESC_BEGIN_INT:.*]] = ptrtoaddr ptr %[[DESC]] to i64
!CHECK: %[[DESC_SIZE:.*]] = sub i64 %[[DESC_END_INT]], %[[DESC_BEGIN_INT]]
!CHECK: %[[DATA_CMP:.*]] = icmp eq ptr %[[ARRAY_OFFSET1]], null
!CHECK: %[[DATA_SIZE:.*]] = select i1 %[[DATA_CMP]], i64 0, i64 %[[COUNT_SEL]]
!CHECK: %[[SEG_CMP:.*]] = icmp eq ptr %[[ARRAY_OFFSET]], null
!CHECK: %[[SEG_SIZE:.*]] = select i1 %[[SEG_CMP]], i64 0, i64 64
!CHECK: %[[ATTACH_CMP:.*]] = icmp eq ptr %[[LOAD_RTTI]], null
!CHECK: %[[ATTACH_SIZE:.*]] = select i1 %[[ATTACH_CMP]], i64 0, i64 8
!CHECK: call void @llvm.memcpy{{.*}}ptr align 8 %.offload_sizes, ptr align 8 @.offload_sizes{{.*}}
!CHECK: %[[BASE0:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: store ptr %[[DESC]], ptr %[[BASE0]], align 8
!CHECK: %[[PTR0:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr %[[DESC]], ptr %[[PTR0]], align 8
!CHECK: %[[SIZE0:.*]] = getelementptr inbounds [7 x i64], ptr %.offload_sizes, i32 0, i32 0
!CHECK: store i64 %[[DESC_SIZE]], ptr %[[SIZE0]], align 8
!CHECK: %[[BASE1:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
!CHECK: store ptr %[[DESC]], ptr %[[BASE1]], align 8
!CHECK: %[[PTR1:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 1
!CHECK: store ptr %[[DESC]], ptr %[[PTR1]], align 8
!CHECK: %[[BASE2:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
!CHECK: store ptr %[[DESC]], ptr %[[BASE2]], align 8
!CHECK: %[[PTR2:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 2
!CHECK: store ptr %[[ARRAY_OFFSET1]], ptr %[[PTR2]], align 8
!CHECK: %[[SIZE2:.*]] = getelementptr inbounds [7 x i64], ptr %.offload_sizes, i32 0, i32 2
!CHECK: store i64 %[[DATA_SIZE]], ptr %[[SIZE2]], align 8
!CHECK: %[[BASE3:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 3
!CHECK: store ptr %[[RES:.*]], ptr %[[BASE3]], align 8
!CHECK: %[[PTR3:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 3
!CHECK: store ptr %[[RES]], ptr %[[PTR3]], align 8
!CHECK: %[[BASE4:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 4
!CHECK: store ptr %[[DESC]], ptr %[[BASE4]], align 8
!CHECK: %[[PTR4:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 4
!CHECK: store ptr %[[ARRAY_OFFSET]], ptr %[[PTR4]], align 8
!CHECK: %[[SIZE4:.*]] = getelementptr inbounds [7 x i64], ptr %.offload_sizes, i32 0, i32 4
!CHECK: store i64 %[[SEG_SIZE]], ptr %[[SIZE4]], align 8
!CHECK: %[[BASE5:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 5
!CHECK: store ptr %[[RTTI_FIELD]], ptr %[[BASE5]], align 8
!CHECK: %[[PTR5:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 5
!CHECK: store ptr %[[LOAD_RTTI]], ptr %[[PTR5]], align 8
!CHECK: %[[SIZE5:.*]] = getelementptr inbounds [7 x i64], ptr %.offload_sizes, i32 0, i32 5
!CHECK: store i64 %[[ATTACH_SIZE]], ptr %[[SIZE5]], align 8
!CHECK: %[[BASE6:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 6
!CHECK: store ptr null, ptr %[[BASE6]], align 8
!CHECK: %[[PTR6:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 6
!CHECK: store ptr null, ptr %[[PTR6]], align 8

!CHECK-LABEL: define {{.*}} @{{.*}}maptype_polymorphic_array_whole_explicit_{{.*}}
!CHECK: %.offload_baseptrs = alloca [7 x ptr], align 8
!CHECK: %.offload_ptrs = alloca [7 x ptr], align 8
!CHECK: %.offload_mappers = alloca [7 x ptr], align 8
!CHECK: %.offload_sizes = alloca [7 x i64], align 8
!CHECK: %[[EXTENT_ACCESS:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %{{.*}}, i32 0, i32 7, i64 0, i32 1
!CHECK: %[[EXTENT:.*]] = load i64, ptr %[[EXTENT_ACCESS]], align 8
!CHECK: %[[BASE_ADDR_FIELD:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %[[DESC:.*]], i32 0, i32 0
!CHECK: %[[ELEM_LEN_FIELD:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %{{.*}}, i32 0, i32 1
!CHECK: %[[ELEM_LEN:.*]] = load i64, ptr %[[ELEM_LEN_FIELD]], align 8
!CHECK: %[[EXTENT_BYTES:.*]] = mul i64 %[[EXTENT]], %[[ELEM_LEN]]
!CHECK: %[[UB_BYTES:.*]] = sub i64 %[[EXTENT_BYTES]], 1
!CHECK: %[[ELEM_LEN_FIELD2:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %{{.*}}, i32 0, i32 1
!CHECK: %[[ELEM_LEN2:.*]] = load i64, ptr %[[ELEM_LEN_FIELD2]], align 8
!CHECK: %[[EXTENT_BYTES2:.*]] = mul i64 %[[EXTENT]], %[[ELEM_LEN2]]
!CHECK: %[[UB_BYTES2:.*]] = sub i64 %[[EXTENT_BYTES2]], 1
!CHECK: %[[RTTI_FIELD:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %[[DESC]], i32 0, i32 8
!CHECK: %[[COUNT_RANGE:.*]] = sub i64 %[[UB_BYTES]], 0
!CHECK: %[[COUNT_RANGE_1:.*]] = add i64 %[[COUNT_RANGE]], 1
!CHECK: %[[COUNT_MUL:.*]] = mul i64 1, %[[COUNT_RANGE_1]]
!CHECK: %[[ELEMENT_COUNT:.*]] = mul i64 %[[COUNT_MUL]], 1
!CHECK: %[[COUNT_CMP:.*]] = icmp eq i64 %[[ELEMENT_COUNT]], 0
!CHECK: %[[COUNT_SEL:.*]] = select i1 %[[COUNT_CMP]], i64 1, i64 %[[ELEMENT_COUNT]]
!CHECK: %[[LOAD_BASE_ADDR:.*]] = load ptr, ptr %[[BASE_ADDR_FIELD]], align 8
!CHECK: %[[ARRAY_OFFSET:.*]] = getelementptr inbounds i8, ptr %[[LOAD_BASE_ADDR]], i64 0
!CHECK: %[[LOAD_RTTI:.*]] = load ptr, ptr %[[RTTI_FIELD]], align 8
!CHECK: %[[LOAD_BASE_ADDR2:.*]] = load ptr, ptr %[[BASE_ADDR_FIELD]], align 8
!CHECK: %[[ARRAY_OFFSET1:.*]] = getelementptr inbounds i8, ptr %[[LOAD_BASE_ADDR2]], i64 0
!CHECK: %[[DESC_END:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %[[DESC]], i32 1
!CHECK: %[[DESC_END_INT:.*]] = ptrtoaddr ptr %[[DESC_END]] to i64
!CHECK: %[[DESC_BEGIN_INT:.*]] = ptrtoaddr ptr %[[DESC]] to i64
!CHECK: %[[DESC_SIZE:.*]] = sub i64 %[[DESC_END_INT]], %[[DESC_BEGIN_INT]]
!CHECK: %[[DATA_CMP:.*]] = icmp eq ptr %[[ARRAY_OFFSET1]], null
!CHECK: %[[DATA_SIZE:.*]] = select i1 %[[DATA_CMP]], i64 0, i64 %[[COUNT_SEL]]
!CHECK: %[[SEG_CMP:.*]] = icmp eq ptr %[[ARRAY_OFFSET]], null
!CHECK: %[[SEG_SIZE:.*]] = select i1 %[[SEG_CMP]], i64 0, i64 64
!CHECK: %[[ATTACH_CMP:.*]] = icmp eq ptr %[[LOAD_RTTI]], null
!CHECK: %[[ATTACH_SIZE:.*]] = select i1 %[[ATTACH_CMP]], i64 0, i64 8
!CHECK: call void @llvm.memcpy{{.*}}ptr align 8 %.offload_sizes, ptr align 8 @.offload_sizes{{.*}}
!CHECK: %[[BASE0:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 0
!CHECK: store ptr %[[DESC]], ptr %[[BASE0]], align 8
!CHECK: %[[PTR0:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 0
!CHECK: store ptr %[[DESC]], ptr %[[PTR0]], align 8
!CHECK: %[[SIZE0:.*]] = getelementptr inbounds [7 x i64], ptr %.offload_sizes, i32 0, i32 0
!CHECK: store i64 %[[DESC_SIZE]], ptr %[[SIZE0]], align 8
!CHECK: %[[BASE1:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 1
!CHECK: store ptr %[[DESC]], ptr %[[BASE1]], align 8
!CHECK: %[[PTR1:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 1
!CHECK: store ptr %[[DESC]], ptr %[[PTR1]], align 8
!CHECK: %[[BASE2:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 2
!CHECK: store ptr %[[DESC]], ptr %[[BASE2]], align 8
!CHECK: %[[PTR2:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 2
!CHECK: store ptr %[[ARRAY_OFFSET1]], ptr %[[PTR2]], align 8
!CHECK: %[[SIZE2:.*]] = getelementptr inbounds [7 x i64], ptr %.offload_sizes, i32 0, i32 2
!CHECK: store i64 %[[DATA_SIZE]], ptr %[[SIZE2]], align 8
!CHECK: %[[BASE3:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 3
!CHECK: store ptr %[[RES:.*]], ptr %[[BASE3]], align 8
!CHECK: %[[PTR3:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 3
!CHECK: store ptr %[[RES]], ptr %[[PTR3]], align 8
!CHECK: %[[BASE4:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 4
!CHECK: store ptr %[[DESC]], ptr %[[BASE4]], align 8
!CHECK: %[[PTR4:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 4
!CHECK: store ptr %[[ARRAY_OFFSET]], ptr %[[PTR4]], align 8
!CHECK: %[[SIZE4:.*]] = getelementptr inbounds [7 x i64], ptr %.offload_sizes, i32 0, i32 4
!CHECK: store i64 %[[SEG_SIZE]], ptr %[[SIZE4]], align 8
!CHECK: %[[BASE5:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 5
!CHECK: store ptr %[[RTTI_FIELD]], ptr %[[BASE5]], align 8
!CHECK: %[[PTR5:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 5
!CHECK: store ptr %[[LOAD_RTTI]], ptr %[[PTR5]], align 8
!CHECK: %[[SIZE5:.*]] = getelementptr inbounds [7 x i64], ptr %.offload_sizes, i32 0, i32 5
!CHECK: store i64 %[[ATTACH_SIZE]], ptr %[[SIZE5]], align 8
!CHECK: %[[BASE6:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_baseptrs, i32 0, i32 6
!CHECK: store ptr null, ptr %[[BASE6]], align 8
!CHECK: %[[PTR6:.*]] = getelementptr inbounds [7 x ptr], ptr %.offload_ptrs, i32 0, i32 6
!CHECK: store ptr null, ptr %[[PTR6]], align 8

!CHECK-LABEL: define internal void @.omp_mapper.{{.*}}wrapper_t_omp_default_mapper
!CHECK: %[[ITEM_DESC:.*]] = getelementptr %{{.*}}Twrapper_t, ptr %{{.*}}, i32 0, i32 1
!CHECK: %[[ELEM_LEN_FIELD:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %{{.*}}, i32 0, i32 1
!CHECK: %[[ELEM_LEN:.*]] = load i64, ptr %[[ELEM_LEN_FIELD]], align 8
!CHECK: %[[DYNAMIC_ITEM_SIZE:.*]] = select i1 %{{.*}}, i64 0, i64 %{{.*}}
!CHECK: call void @.omp_mapper.{{.*}}base_t_omp_default_mapper(ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, i64 %[[DYNAMIC_ITEM_SIZE]], i64 %{{.*}}, ptr {{.*}}) {{.*}}
