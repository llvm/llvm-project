! RUN: %flang -ffast-amd-memory-allocator -S -emit-llvm -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa --offload-arch=gfx90a -o - %s | FileCheck %s --check-prefix=CHECK-OMP
! RUN: %flang -ffast-amd-memory-allocator -S -emit-llvm -target amdgcn-- -o - %s | FileCheck %s --check-prefix=CHECK

subroutine func_t_device()
  !$omp declare target enter(func_t_device) device_type(nohost)
    integer, ALLOCATABLE :: poly

! CHECK-OMP-NOT: call void @_FortranAAMDAllocatableSetAllocIdx({{.*}}, i32 1)
! CHECK: call void @_FortranAAMDAllocatableSetAllocIdx({{.*}}, i32 1)
! CHECK-OMP: call i32 @_FortranAAllocatableAllocate
! CHECK: call i32 @_FortranAAllocatableAllocate   
    ALLOCATE(poly)

! CHECK-OMP: call i32 @_FortranAAllocatableDeallocate
! CHECK: call i32 @_FortranAAllocatableDeallocate
    DEALLOCATE(poly)
end subroutine func_t_device

program main
  implicit none
  !$omp target
    call func_t_device()
  !$omp end target
end program
