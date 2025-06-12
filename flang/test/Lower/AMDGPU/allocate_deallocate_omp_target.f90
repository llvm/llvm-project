! RUN: %flang -ffast-amd-memory-allocator -S -emit-llvm -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa --offload-arch=gfx90a -o - %s | FileCheck %s --check-prefix=CHECK-OMP
! RUN: %flang -ffast-amd-memory-allocator -S -emit-llvm -target amdgcn-- -o - %s | FileCheck %s --check-prefix=CHECK

program main
   implicit none
   !$omp requires unified_shared_memory
   REAL, DIMENSION(:), ALLOCATABLE :: poly
   integer,parameter :: n = 10
   integer :: i,j
     !$omp target teams distribute parallel do private(poly)
     do j=1,n

! CHECK-OMP-NOT: call void @_FortranAAMDAllocatableSetAllocIdx({{.*}}, i32 1)
! CHECK: call void @_FortranAAMDAllocatableSetAllocIdx({{.*}}, i32 1)
! CHECK-OMP: call i32 @_FortranAAllocatableAllocate
! CHECK: call i32 @_FortranAAllocatableAllocate
       ALLOCATE(poly(1:3))
       poly = 2.0_8
! CHECK-OMP: call i32 @_FortranAAllocatableDeallocate
! CHECK: call i32 @_FortranAAllocatableDeallocate
       DEALLOCATE(poly)
     enddo
     !$omp end target teams distribute parallel do
end program
