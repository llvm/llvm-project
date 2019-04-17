#ifndef LLVM_CLANG_TEST_OPENMP_INPUTS_DECLARE_SIMD_FIX_H
#define LLVM_CLANG_TEST_OPENMP_INPUTS_DECLARE_SIMD_FIX_H

#pragma omp declare simd
float foo(float a, float b, int c);
float bar(float a, float b, int c);

#endif
