// RUN: %clang_cc1 -fsyntax-only -fopenmp-simd -fopenmp-version=51 -verify=expected,omp51,omp -fno-openmp-extensions %s
// RUN: %clang_cc1 -fsyntax-only -fopenmp -fopenmp-version=51 -verify=expected,omp51,omp -fno-openmp-extensions -Wno-source-uses-openmp %s

#include "Inputs/nesting_of_regions.cpp"
