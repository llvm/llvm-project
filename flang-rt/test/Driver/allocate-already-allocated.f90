! UNSUPPORTED: offload-cuda

! RUN: %flang %isysroot -L"%libdir" %s -o %t
! RUN: not --crash env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t 2>&1 \
! RUN:   | FileCheck %s

! CHECK: fatal Fortran runtime error({{.*}}allocate-already-allocated.f90:{{[0-9]+}}): The object 'array' is already allocated

program allocate_twice
  integer, allocatable :: array(:)

  allocate(array(5))
  allocate(array(8))
end program
