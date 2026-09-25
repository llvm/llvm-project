! RUN: %flang_fc1 -fopenmp-default-allocate=target -mmlir -use-alloc-runtime -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s --check-prefix=CHECK-OMP
! RUN: %flang_fc1 -fopenmp-default-allocate=target -mmlir -use-alloc-runtime -emit-hlfir -triple amdgcn-amd-amdhsa %s -o - | FileCheck %s --check-prefix=CHECK
subroutine func_t_device()
  !$omp declare target enter(func_t_device) device_type(nohost)
    integer, ALLOCATABLE :: poly

! CHECK-OMP-NOT: fir.call @_FortranAOpenMPAllocatableSetAllocIdx
! CHECK: fir.call @_FortranAOpenMPAllocatableSetAllocIdx
! CHECK-OMP: fir.call @_FortranAAllocatableAllocate
! CHECK: fir.call @_FortranAAllocatableAllocate
    ALLOCATE(poly)

! CHECK-OMP: fir.call @_FortranAAllocatableDeallocate
! CHECK: fir.call @_FortranAAllocatableDeallocate
    DEALLOCATE(poly)
end subroutine func_t_device

program main
  implicit none
  !$omp target
    call func_t_device()
  !$omp end target
end program
