! Verify that declarative ALLOCATE on derived-type variables that require
! default initialization or finalization emits a lowering warning and does
! not generate omp.allocate_dir / omp.allocate_free for those variables.

! RUN: %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=51 %s -o - 2>&1 | FileCheck %s

module derived_allocate_warning
  type :: init_type
    integer :: k = 42
  end type

  type :: final_type
    integer :: a
  contains
    final :: final_type_cleanup
  end type

contains
  subroutine final_type_cleanup(this)
    type(final_type) :: this
  end subroutine

  subroutine init_allocate_warning
    type(init_type) :: obj

    !$omp allocate(obj) allocator(1)
  end subroutine init_allocate_warning

  subroutine final_allocate_warning
    type(final_type) :: obj

    !$omp allocate(obj) allocator(1)
  end subroutine final_allocate_warning
end module derived_allocate_warning

! Warnings are emitted during lowering before HLFIR is printed.
! CHECK: warning: {{.*}}TODO : OpenMP declarative ALLOCATE on derived-type variables with initialization or finalization is not yet supported, ignoring the ALLOCATE directive for 'obj'
! CHECK: warning: {{.*}}TODO : OpenMP declarative ALLOCATE on derived-type variables with initialization or finalization is not yet supported, ignoring the ALLOCATE directive for 'obj'

! CHECK-LABEL: func.func @_QMderived_allocate_warningPinit_allocate_warning
! CHECK-NOT: omp.allocate_dir
! CHECK-NOT: omp.allocate_free

! CHECK-LABEL: func.func @_QMderived_allocate_warningPfinal_allocate_warning
! CHECK-NOT: omp.allocate_dir
! CHECK-NOT: omp.allocate_free
