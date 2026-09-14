! UNSUPPORTED: offload-cuda

! RUN: %flang %isysroot -L"%libdir" %s -o %t
! RUN: not --crash env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t

! Double ALLOCATE of the same object must abort. The assert message is
! checked in Lower/allocate-already-allocated.f90.

program allocate_twice
  integer, allocatable :: array(:)

  allocate(array(5))
  allocate(array(8))
end program
