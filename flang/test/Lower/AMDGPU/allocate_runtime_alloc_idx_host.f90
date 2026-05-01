! RUN: %flang -fopenmp-default-allocate=host -S -emit-llvm --offload-targets=amdgcn-amd-amdhsa -o - %s 2>&1 | FileCheck %s

! Verify that host mode does not insert OpenMPAllocatableSetAllocIdx calls.

! CHECK-NOT: call void @_FortranAOpenMPAllocatableSetAllocIdx

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