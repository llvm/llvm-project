! RUN: %flang -target amdgcn-- -S -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK-AMDGPU
! RUN: %flang -target x86_64-- -S -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK-x86_64

subroutine allocate_deallocate()
  real, allocatable :: x

  allocate(x)
! CHECK-AMDGPU: call i32 @_FortranAAllocatableAllocate
! CHECK-x86_64: call ptr @malloc

  deallocate(x)
! CHECK-AMDGPU: call i32 @_FortranAAllocatableDeallocate
! CHECK-x86_64: call void @free
end subroutine

subroutine allocate_deallocate_ptr()
  integer, pointer :: x

  allocate(x)
! CHECK-AMDGPU: call i32 @_FortranAPointerAllocate
! CHECK-x86_64: call i32 @_FortranAPointerAllocate

  deallocate(x)
! CHECK-AMDGPU: call i32 @_FortranAPointerDeallocate
! CHECK-x86_64: call i32 @_FortranAPointerDeallocate
end subroutine
