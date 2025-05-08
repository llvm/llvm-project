! RUN: %flang -target amdgcn-- -ffast-amd-memory-allocator -mmlir -use-alloc-runtime -S -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK
subroutine allocate_deallocate()
  real, allocatable :: x
! CHECK: call void @_FortranAAMDAllocatableSetAllocIdx({{.*}}, i32 1)
! CHECK: call i32 @_FortranAAllocatableAllocate
  allocate(x)

! CHECK: call i32 @_FortranAAllocatableDeallocate
  deallocate(x)
end subroutine

subroutine test_allocatable_scalar(a)
  real, save, allocatable :: x1, x2
  real :: a

! CHECK: call void @_FortranAAMDAllocatableSetAllocIdx({{.*}}, i32 1)
! CHECK: call i32 @_FortranAAllocatableAllocateSource
  allocate(x1, x2, source = a)
end
