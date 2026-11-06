!RUN: %flang_fc1 -fopenmp -fopenmp-default-allocate=target -emit-fir %s -o - | FileCheck %s

program omp_alloc_init
    !CHECK: fir.call @_FortranAOpenMPRegisterAllocator()
end program omp_alloc_init
