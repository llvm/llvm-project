! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

! Verify that the inlined allocation path checks the allocation status before
! allocating. The check is on by default. See
! allocate-already-allocated-disabled.f90 for -fno-check-already-allocated.

! CHECK-LABEL: func.func @_QPallocate_twice()
subroutine allocate_twice()
  integer, allocatable :: array(:)

  ! CHECK: %[[IS_NOT_ALLOCATED_1:.*]] = arith.cmpi eq
  ! CHECK: fir.assert %[[IS_NOT_ALLOCATED_1]], status, "The object 'array' is already allocated"
  ! CHECK: fir.allocmem
  allocate(array(5))

  ! CHECK: %[[IS_NOT_ALLOCATED_2:.*]] = arith.cmpi eq
  ! CHECK: fir.assert %[[IS_NOT_ALLOCATED_2]], status, "The object 'array' is already allocated"
  ! CHECK: fir.allocmem
  allocate(array(8))
end subroutine
