!RUN: %flang_fc1 -fopenmp -ffast-amd-memory-allocator -emit-fir %s -o - | FileCheck %s

program amd_alloc_init
    !CHECK: fir.call @_FortranAAMDRegisterAllocator()
end program amd_alloc_init