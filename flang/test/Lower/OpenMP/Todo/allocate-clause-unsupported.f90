! RUN: split-file %s %t
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=51 -o - %t/duplicate.f90 2>&1 | FileCheck %s --check-prefix=DUPLICATE
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=51 -o - %t/array.f90 2>&1 | FileCheck %s --check-prefix=ARRAY
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=51 -o - %t/derived.f90 2>&1 | FileCheck %s --check-prefix=DERIVED
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=51 -o - %t/pointer.f90 2>&1 | FileCheck %s --check-prefix=POINTER
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=51 -o - %t/allocatable.f90 2>&1 | FileCheck %s --check-prefix=ALLOCATABLE
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=51 -o - %t/assumed-length.f90 2>&1 | FileCheck %s --check-prefix=ASSUMED-LENGTH

! DUPLICATE: not yet implemented: ALLOCATE clause item appears more than once
! ARRAY: not yet implemented: ALLOCATE clause currently supports only fixed-size intrinsic scalar PRIVATE or FIRSTPRIVATE items
! DERIVED: not yet implemented: ALLOCATE clause currently supports only fixed-size intrinsic scalar PRIVATE or FIRSTPRIVATE items
! POINTER: not yet implemented: ALLOCATE clause currently supports only fixed-size intrinsic scalar PRIVATE or FIRSTPRIVATE items
! ALLOCATABLE: not yet implemented: ALLOCATE clause currently supports only fixed-size intrinsic scalar PRIVATE or FIRSTPRIVATE items
! ASSUMED-LENGTH: not yet implemented: ALLOCATE clause currently supports only fixed-size intrinsic scalar PRIVATE or FIRSTPRIVATE items

!--- duplicate.f90
subroutine duplicate(x)
  integer :: x
  !$omp parallel private(x) allocate(x) allocate(x)
    x = 1
  !$omp end parallel
end subroutine

!--- pointer.f90
subroutine pointer(x)
  integer, pointer :: x
  !$omp parallel private(x) allocate(x)
    x = 1
  !$omp end parallel
end subroutine

!--- allocatable.f90
subroutine allocatable(x)
  integer, allocatable :: x
  !$omp parallel private(x) allocate(x)
    x = 1
  !$omp end parallel
end subroutine

!--- assumed-length.f90
subroutine assumed_length(x)
  character(*) :: x
  !$omp parallel private(x) allocate(x)
    x = "test"
  !$omp end parallel
end subroutine

!--- array.f90
subroutine array(x)
  integer :: x(4)
  !$omp parallel private(x) allocate(x)
    x = 1
  !$omp end parallel
end subroutine

!--- derived.f90
subroutine derived()
  type t
    integer :: value
  end type
  type(t) :: x
  !$omp parallel private(x) allocate(x)
    x%value = 1
  !$omp end parallel
end subroutine
