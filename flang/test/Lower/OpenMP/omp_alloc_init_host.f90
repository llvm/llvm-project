!RUN: %flang_fc1 -fopenmp -fopenmp-default-allocate=host -emit-fir %s -o - | FileCheck %s

program omp_alloc_init_host
    !CHECK-NOT: fir.call @_FortranAOpenMPRegisterAllocator()
end program omp_alloc_init_host
