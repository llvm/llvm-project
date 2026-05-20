! Test lowering of COPYIN clause for derived types with allocatable components.
! Threadprivate copies may have uninitialized allocatable component descriptors
! (zero-filled by the OpenMP runtime). For non-master threads, use
! temporary_lhs semantics so that AssignTemporary initializes descriptors
! before performing the deep copy. The master thread is skipped via a runtime
! address comparison (lhs == rhs means same threadprivate storage).

! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPcopyin_derived_alloc_comp
subroutine copyin_derived_alloc_comp()
  type inner
    integer, allocatable :: a(:)
  end type inner
  type outer
    type(inner) :: s
  end type outer
  type(outer), save :: x(2)
  !$omp threadprivate(x)

  allocate(x(1)%s%a(3))
  x(1)%s%a = 42

! CHECK:      omp.parallel {
! CHECK:        omp.threadprivate
! CHECK:        hlfir.declare
! CHECK:        fir.convert {{.*}} -> index
! CHECK:        fir.convert {{.*}} -> index
! CHECK:        arith.cmpi eq, {{.*}} : index
! CHECK:        fir.if
! CHECK:        } else {
! CHECK:          hlfir.assign {{.*}} temporary_lhs
! CHECK:        }
! CHECK:        omp.barrier
  !$omp parallel copyin(x)
    call sub(x(1)%s%a)
  !$omp end parallel
  deallocate(x(1)%s%a)
end subroutine

! CHECK-LABEL: func.func @_QPcopyin_scalar_derived_alloc_comp
subroutine copyin_scalar_derived_alloc_comp()
  type dt
    integer, allocatable :: a(:)
  end type dt
  type(dt), save :: y
  !$omp threadprivate(y)
  allocate(y%a(5))
  y%a = 10

! CHECK:      omp.parallel {
! CHECK:        omp.threadprivate
! CHECK:        hlfir.declare
! CHECK:        fir.convert {{.*}} -> index
! CHECK:        fir.convert {{.*}} -> index
! CHECK:        arith.cmpi eq, {{.*}} : index
! CHECK:        fir.if
! CHECK:        } else {
! CHECK:          hlfir.assign {{.*}} temporary_lhs
! CHECK:        }
! CHECK:        omp.barrier
  !$omp parallel copyin(y)
    call sub(y%a)
  !$omp end parallel
  deallocate(y%a)
end subroutine

! Derived type WITHOUT allocatable components: plain assign, no address check.
! CHECK-LABEL: func.func @_QPcopyin_no_alloc_comp
subroutine copyin_no_alloc_comp()
  type simple
    integer :: val
  end type simple
  type(simple), save :: z
  !$omp threadprivate(z)
! CHECK:      omp.parallel {
! CHECK:        omp.threadprivate
! CHECK:        hlfir.declare
! CHECK-NOT:    arith.cmpi
! CHECK:        hlfir.assign
! CHECK-NOT:    temporary_lhs
! CHECK:        omp.barrier
  !$omp parallel copyin(z)
    call sub2(z%val)
  !$omp end parallel
end subroutine

