! REQUIRES: openmp_runtime
! RUN: %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=51 -o - %s 2>&1 | FileCheck %s --check-prefix=HLFIR

subroutine allocator_omitted(x, y)
  integer :: x, y
  !$omp parallel private(x, y) allocate(x)
    x = 1
    y = 2
  !$omp end parallel
end subroutine

! HLFIR-LABEL: func.func @_QPallocator_omitted
! HLFIR: %[[NULL_ALLOC:.*]] = arith.constant 0 : i32
! HLFIR: omp.parallel allocate(%[[NULL_ALLOC]] : i32 -> %[[X:.*]]#0 : !fir.ref<i32>)
! HLFIR-SAME: private({{.*}} %[[X]]#0 -> %[[X_PRIVATE:.*]], {{.*}} -> %[[Y_PRIVATE:.*]] : !fir.ref<i32>, !fir.ref<i32>) {
! HLFIR: } {allocate_private_indices = array<i64: 0>}

subroutine allocator_explicit(x)
  use omp_lib
  integer :: x

  !$omp parallel private(x) allocate(allocator(omp_null_allocator): x)
    x = 1
  !$omp end parallel

  !$omp parallel private(x) allocate(allocator(omp_default_mem_alloc): x)
    x = 2
  !$omp end parallel
end subroutine

! HLFIR-LABEL: func.func @_QPallocator_explicit
! HLFIR: omp.parallel allocate(%c0_i64 : i64 -> %[[X:.*]]#0 : !fir.ref<i32>)
! HLFIR-SAME: private({{.*}} %[[X]]#0 -> {{.*}} : !fir.ref<i32>) {
! HLFIR: omp.parallel allocate(%c1_i64 : i64 -> %[[X2:.*]]#0 : !fir.ref<i32>)
! HLFIR-SAME: private({{.*}} %[[X2]]#0 -> {{.*}} : !fir.ref<i32>) {

subroutine allocator_dynamic(x, allocator)
  use omp_lib, only : omp_allocator_handle_kind
  integer :: x
  integer(kind=omp_allocator_handle_kind), intent(in) :: allocator
  !$omp parallel firstprivate(x) allocate(allocator(allocator): x)
    x = x + 1
  !$omp end parallel
end subroutine

! HLFIR-LABEL: func.func @_QPallocator_dynamic
! HLFIR: %[[ALLOCATOR:.*]] = fir.load
! HLFIR: omp.parallel allocate(%[[ALLOCATOR]] : i64 -> %[[X:.*]]#0 : !fir.ref<i32>)
! HLFIR-SAME: private({{.*}} %[[X]]#0 -> %[[X_PRIVATE:.*]] : !fir.ref<i32>) {
! HLFIR: } {allocate_private_indices = array<i64: 0>}

function allocator_value() result(allocator)
  use omp_lib, only : omp_allocator_handle_kind, omp_default_mem_alloc
  integer(kind=omp_allocator_handle_kind) :: allocator
  allocator = omp_default_mem_alloc
end function

subroutine allocator_expression(x, y)
  use omp_lib, only : omp_allocator_handle_kind
  integer :: x, y
  integer(kind=omp_allocator_handle_kind), external :: allocator_value
  !$omp parallel private(x, y) allocate(allocator(allocator_value()): x, y)
    x = 1
    y = 2
  !$omp end parallel
end subroutine

! HLFIR-LABEL: func.func @_QPallocator_expression
! HLFIR-COUNT-1: fir.call @_QPallocator_value()
! HLFIR: omp.parallel allocate(%[[ALLOCATOR:.*]] : i64 -> %{{.*}}#0 : !fir.ref<i32>, %[[ALLOCATOR]] : i64 -> %{{.*}}#0 : !fir.ref<i32>)
! HLFIR: } {allocate_private_indices = array<i64: 0, 1>}

subroutine allocator_host_associated()
  integer :: x
contains
  subroutine inner()
    !$omp parallel private(x) allocate(x)
      x = 1
    !$omp end parallel
  end subroutine
end subroutine

! HLFIR-LABEL: func.func private @_QFallocator_host_associatedPinner
! HLFIR: omp.parallel allocate({{.*}} -> %[[X:.*]]#0 : !fir.ref<i32>)
! HLFIR-SAME: private({{.*}} %[[X]]#0 -> {{.*}} : !fir.ref<i32>) {
! HLFIR: } {allocate_private_indices = array<i64: 0>}

subroutine allocator_order(x, y, z)
  use omp_lib
  integer :: x, y, z
  !$omp parallel private(x, y, z) &
  !$omp& allocate(allocator(omp_high_bw_mem_alloc): y, x) &
  !$omp& allocate(allocator(omp_low_lat_mem_alloc): z)
    x = 1
    y = 2
    z = 3
  !$omp end parallel
end subroutine

! HLFIR-LABEL: func.func @_QPallocator_order
! HLFIR: omp.parallel allocate(
! HLFIR-SAME: private(
! HLFIR: } {allocate_private_indices = array<i64: 1, 0, 2>}

subroutine allocator_scalar_types(r, c, l, s)
  real :: r
  complex :: c
  logical :: l
  character(4) :: s
  !$omp parallel private(r, c, l, s) allocate(r, c, l, s)
    r = 1.0
    c = (1.0, 2.0)
    l = .true.
    s = "test"
  !$omp end parallel
end subroutine

! HLFIR-LABEL: func.func @_QPallocator_scalar_types
! HLFIR: } {allocate_private_indices = array<i64: 0, 1, 2, 3>}
