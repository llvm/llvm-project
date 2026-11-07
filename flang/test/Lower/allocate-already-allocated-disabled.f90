! RUN: bbc -emit-hlfir -fno-check-already-allocated %s -o - | FileCheck %s

! With -fno-check-already-allocated, the inlined allocation path does not emit
! the allocation status check.

! CHECK-LABEL: func.func @_QPallocate_twice()
subroutine allocate_twice()
  integer, allocatable :: array(:)

  ! CHECK-NOT: fir.assert
  ! CHECK: fir.allocmem
  allocate(array(5))

  ! CHECK-NOT: fir.assert
  ! CHECK: fir.allocmem
  allocate(array(8))
  ! CHECK-NOT: fir.assert
end subroutine
