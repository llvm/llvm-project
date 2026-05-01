! RUN: %flang -fopenmp-default-allocate=target -S -emit-llvm --offload-targets=amdgcn-amd-amdhsa -o - %s | FileCheck %s

subroutine allocate_deallocate()
  real, allocatable :: x
! CHECK: call void @_FortranAOpenMPAllocatableSetAllocIdx({{.*}}, i32 1)
! CHECK: call i32 @_FortranAAllocatableAllocate
  allocate(x)

! CHECK: call i32 @_FortranAAllocatableDeallocate
  deallocate(x)
end subroutine

subroutine test_allocatable_scalar(a)
  real, save, allocatable :: x1, x2
  real :: a

! CHECK: call void @_FortranAOpenMPAllocatableSetAllocIdx({{.*}}, i32 1)
! CHECK: call i32 @_FortranAAllocatableAllocateSource
  allocate(x1, x2, source = a)
end
