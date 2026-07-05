! Test that derived type allocatable members of private copies are properly
! initialized.
!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

module m1
  type x
     integer, allocatable :: x1(:)
  end type

  type y
     integer :: y1(10)
  end type

contains

!CHECK-LABEL: omp.private {type = private} @_QMm1Ftest_class_allocatable_array
!CHECK:       fir.call @_FortranAInitialize
!CHECK-NOT:   omp.barrier
!CHECK:       omp.yield

!CHECK-LABEL: omp.private {type = private} @_QMm1Ftest_class_allocatable
!CHECK:       fir.call @_FortranAInitialize
!CHECK-NOT:   omp.barrier
!CHECK:       omp.yield

!CHECK-LABEL: omp.private {type = private} @_QMm1Ftest_allocatable
!CHECK:       fir.call @_FortranAInitialize
!CHECK-NOT:   omp.barrier
!CHECK:       omp.yield

!CHECK-LABEL: omp.private {type = private} @_QMm1Ftest_pointer
!CHECK-NOT:   fir.call @_FortranAInitializeClone
!CHECK-NOT:   omp.barrier
!CHECK:       omp.yield

!CHECK-LABEL: omp.private {type = private} @_QMm1Ftest_nested
!CHECK:       fir.call @_FortranAInitializeClone
!CHECK-NOT:   omp.barrier
!CHECK:       omp.yield

!CHECK-LABEL: omp.private {type = private} @_QMm1Ftest_array_of_allocs
!CHECK:       fir.call @_FortranAInitializeClone
!CHECK-NOT:   omp.barrier
!CHECK:       omp.yield
!CHECK:       } dealloc {
!CHECK:       fir.call @_FortranAAllocatableDeallocate
!CHECK:       omp.yield

!CHECK-LABEL: omp.private {type = firstprivate} @_QMm1Ftest_array
!CHECK:       fir.call @_FortranAInitialize(
!CHECK-NOT:   fir.call @_FortranAInitializeClone
!CHECK-NOT:   omp.barrier
!CHECK:       omp.yield

!CHECK-LABEL: omp.private {type = private} @_QMm1Ftest_array
!CHECK:       fir.call @_FortranAInitialize(
!CHECK:       fir.call @_FortranAInitializeClone
!CHECK-NOT:   omp.barrier
!CHECK:       omp.yield

!CHECK-LABEL: omp.private {type = private} @_QMm1Ftest_scalar
!CHECK:       fir.call @_FortranAInitializeClone
!CHECK-NOT:   omp.barrier
!CHECK:       omp.yield

  subroutine test_scalar()
    type(x) :: v
    allocate(v%x1(5))

    !$omp parallel private(v)
    !$omp end parallel
  end subroutine

! Test omp sections lastprivate(v, v2)
! - InitializeClone must not be called for v2, that doesn't have an
!   allocatable member.
! - InitializeClone must be called for v, that has an allocatable member.
! - To avoid race conditions between InitializeClone and lastprivate, a
!   barrier must be present after the initializations.
!CHECK-LABEL: func @_QMm1Ptest_array
!CHECK:       fir.call @_FortranAInitializeClone
!CHECK-NEXT:  omp.barrier
  subroutine test_array()
    type(x) :: v(10)
    type(y) :: v2(10)
    allocate(v(1)%x1(5))

    !$omp parallel private(v)
    !$omp end parallel

    !$omp parallel
      !$omp sections lastprivate(v2, v)
      !$omp end sections
    !$omp end parallel

    !$omp parallel firstprivate(v)
    !$omp end parallel
  end subroutine

  subroutine test_array_of_allocs()
    type(x), allocatable  :: v(:)
    allocate(v(10))
    allocate(v(1)%x1(5))

    !$omp parallel private(v)
    !$omp end parallel
  end subroutine

  subroutine test_nested()
    type dt1
      integer, allocatable :: a(:)
    end type

    type dt2
      type(dt1) :: d1
    end type

    type(dt2) :: d2
    allocate(d2%d1%a(10))

    !$omp parallel private(d2)
    !$omp end parallel
  end subroutine

  subroutine test_pointer()
    type(x), pointer :: ptr

    !$omp parallel private(ptr)
    !$omp end parallel
  end subroutine

  subroutine test_allocatable()
    type needs_init
      integer :: i = 1
    end type
    type(needs_init), allocatable :: a

    !$omp parallel private(a)
    !$omp end parallel
  end subroutine

  subroutine test_class_allocatable()
    type needs_init
      integer :: i = 1
    end type
    class(needs_init), allocatable :: a

    !$omp parallel private(a)
    !$omp end parallel
  end subroutine

  subroutine test_class_allocatable_array()
    type needs_init
      integer :: i = 1
    end type
    class(needs_init), allocatable :: a(:)

    !$omp parallel private(a)
    !$omp end parallel
  end subroutine
end module
