! RUN: %flang_fc1 -fopenmp-default-allocate=host -emit-hlfir %s -o - | FileCheck %s

! Verify that host mode does not insert OpenMPAllocatableSetAllocIdx calls.

! CHECK-NOT: fir.call @_FortranAOpenMPAllocatableSetAllocIdx

subroutine allocate_deallocate()
  real, allocatable :: x
  allocate(x)
  deallocate(x)
end subroutine

subroutine test_allocatable_scalar(a)
  real, save, allocatable :: x1, x2
  real :: a
  allocate(x1, x2, source = a)
end