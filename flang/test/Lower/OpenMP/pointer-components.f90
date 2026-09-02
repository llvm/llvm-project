! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s

! Test 1: POINTER-only derieved type should not trigger implicit mapper 
subroutine test_pointer_only
  type :: dt_pointer
    real, pointer :: p(:)
  end type

  type(dt_pointer) :: obj

  ! CHECK-NOT: omp.declare_mapper @{{.*}}dt_pointer
  !$omp target map(tofrom: obj)
  !$omp end target
end subroutine test_pointer_only

! Test 2: POINTER to derieved type with ALLOCATABLE must not trigger mapper 
subroutine test_pointer_to_allocatable_dt
  type :: inner_dt
    real, allocatable :: data(:)
  end type

  type :: outer_dt
    type(inner_dt), pointer :: nested_ptr
  end type

  type(outer_dt) :: obj

  ! CHECK-NOT: omp.declare_mapper @{{.*}}outer_dt
  !$omp target map(tofrom: obj)
  !$omp end target
end subroutine test_pointer_to_allocatable_dt

! Test 3: ALLOCATABLE member must still trigger mapper
subroutine test_allocatable_still_triggers_mapper
  type :: alloc_dt
    real, allocatable :: a(:)
  end type

  type(alloc_dt) :: obj
  allocate(obj%a(4))

  ! CHECK: omp.declare_mapper @{{.*}}alloc_dt
  !$omp target map(tofrom: obj)
    obj%a(1) = 1.0
  !$omp end target
end subroutine test_allocatable_still_triggers_mapper