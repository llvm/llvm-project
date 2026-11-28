! RUN: %flang_fc1 -fopenmp-default-allocate=target -mmlir -use-alloc-runtime -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s --check-prefix=CHECK-OMP
! RUN: %flang_fc1 -fopenmp-default-allocate=target -mmlir -use-alloc-runtime -emit-hlfir -triple amdgcn-amd-amdhsa %s -o - | FileCheck %s --check-prefix=CHECK
program main
   implicit none
   !$omp requires unified_shared_memory
   REAL, DIMENSION(:), ALLOCATABLE :: poly
   integer,parameter :: n = 10
   integer :: i,j
     !$omp target teams distribute parallel do private(poly)
     do j=1,n

! CHECK-OMP-NOT: fir.call @_FortranAOpenMPAllocatableSetAllocIdx
! CHECK: fir.call @_FortranAOpenMPAllocatableSetAllocIdx
! CHECK-OMP: fir.call @_FortranAAllocatableAllocate
! CHECK: fir.call @_FortranAAllocatableAllocate
       ALLOCATE(poly(1:3))
       poly = 2.0_8
! CHECK-OMP: fir.call @_FortranAAllocatableDeallocate
! CHECK: fir.call @_FortranAAllocatableDeallocate
       DEALLOCATE(poly)
     enddo
     !$omp end target teams distribute parallel do
end program