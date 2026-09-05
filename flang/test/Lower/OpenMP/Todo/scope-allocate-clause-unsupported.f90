! RUN: split-file %s %t
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=52 -o - %t/array.f90 2>&1 | FileCheck %s --check-prefix=ARRAY
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=52 -o - %t/derived.f90 2>&1 | FileCheck %s --check-prefix=DERIVED
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=52 -o - %t/pointer.f90 2>&1 | FileCheck %s --check-prefix=POINTER
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=52 -o - %t/allocatable.f90 2>&1 | FileCheck %s --check-prefix=ALLOCATABLE
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=52 -o - %t/assumed-length.f90 2>&1 | FileCheck %s --check-prefix=ASSUMED-LENGTH
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=52 -o - %t/save.f90 2>&1 | FileCheck %s --check-prefix=SAVE
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=52 -o - %t/common.f90 2>&1 | FileCheck %s --check-prefix=COMMON

! ARRAY: not yet implemented: ALLOCATE clause currently supports only fixed-size intrinsic scalar PRIVATE or FIRSTPRIVATE items
! DERIVED: not yet implemented: ALLOCATE clause currently supports only fixed-size intrinsic scalar PRIVATE or FIRSTPRIVATE items
! POINTER: not yet implemented: ALLOCATE clause currently supports only fixed-size intrinsic scalar PRIVATE or FIRSTPRIVATE items
! ALLOCATABLE: not yet implemented: ALLOCATE clause currently supports only fixed-size intrinsic scalar PRIVATE or FIRSTPRIVATE items
! ASSUMED-LENGTH: not yet implemented: ALLOCATE clause currently supports only fixed-size intrinsic scalar PRIVATE or FIRSTPRIVATE items
! SAVE: not yet implemented: ALLOCATE clause on SCOPE currently does not support SAVE or common block entities
! COMMON: not yet implemented: ALLOCATE clause on SCOPE currently does not support SAVE or common block entities

!--- array.f90
subroutine array(x)
  integer :: x(4)
  !$omp scope private(x) allocate(align(64): x)
    x = 1
  !$omp end scope
end subroutine

!--- derived.f90
subroutine derived()
  type t
    integer :: value
  end type
  type(t) :: x
  !$omp scope private(x) allocate(x)
    x%value = 1
  !$omp end scope
end subroutine

!--- pointer.f90
subroutine pointer(x)
  integer, pointer :: x
  !$omp scope private(x) allocate(x)
    x = 1
  !$omp end scope
end subroutine

!--- allocatable.f90
subroutine allocatable(x)
  integer, allocatable :: x
  !$omp scope private(x) allocate(x)
    x = 1
  !$omp end scope
end subroutine

!--- assumed-length.f90
subroutine assumed_length(x)
  character(*) :: x
  !$omp scope private(x) allocate(x)
    x = "test"
  !$omp end scope
end subroutine

!--- save.f90
subroutine save_entity()
  integer, save :: x
  !$omp scope private(x) allocate(x)
    x = 1
  !$omp end scope
end subroutine

!--- common.f90
subroutine common_block()
  integer :: x
  common /block/ x
  !$omp scope private(x) allocate(x)
    x = 1
  !$omp end scope
end subroutine
