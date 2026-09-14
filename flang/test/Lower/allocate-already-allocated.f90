! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

! Verify that the inlined allocation path checks the allocation status before
! calling the actual allocation function

! CHECK-LABEL: func.func @_QPallocate_twice()
subroutine allocate_twice()
  integer, allocatable :: array(:)

  ! CHECK: %[[IS_NOT_ALLOCATED_1:.*]] = arith.cmpi eq
  ! CHECK: fir.assert %[[IS_NOT_ALLOCATED_1]], "The object 'array' is already allocated"
  ! CHECK: fir.allocmem
  allocate(array(5))

  ! CHECK: %[[IS_NOT_ALLOCATED_2:.*]] = arith.cmpi eq
  ! CHECK: fir.assert %[[IS_NOT_ALLOCATED_2]], "The object 'array' is already allocated"
  ! CHECK: fir.allocmem
  allocate(array(8))
end subroutine
