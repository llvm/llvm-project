! RUN: %flang -target amdgcn-- -mmlir -use-alloc-runtime -S -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK
! RUN: %flang -target amdgcn-- -S -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK-NO-FLAG

! Test to check if usage of flag -use-alloc-runtime results in runtime calls.

subroutine allocate_deallocate()
  real, allocatable :: x

  allocate(x)
! CHECK: call i32 @_FortranAAllocatableAllocate
! CHECK-NO-FLAG: call ptr @malloc

  deallocate(x)
! CHECK: call i32 @_FortranAAllocatableDeallocate
! CHECK-NO-FLAG: call void @free
end subroutine

subroutine allocate_deallocate_ptr()
  integer, pointer :: x

  allocate(x)
! CHECK: call i32 @_FortranAPointerAllocate
! CHECK-NO-FLAG: call i32 @_FortranAPointerAllocate

  deallocate(x)
! CHECK: call i32 @_FortranAPointerDeallocate
! CHECK-NO-FLAG: call i32 @_FortranAPointerDeallocate
end subroutine
