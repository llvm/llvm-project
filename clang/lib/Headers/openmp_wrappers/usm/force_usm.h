#ifndef __CLANG_FORCE_OPENMP_USM
#define __CLANG_FORCE_OPENMP_USM

// Used internally to implement -fopenmp-apu-mode
// Allowing us to enforce requiring USM

#pragma omp requires unified_shared_memory

#endif
