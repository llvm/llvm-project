// RUN: %clang_cc1 -fsyntax-only -fopenmp-simd -fopenmp-version=45 -fno-openmp-extensions -verify=expected,omp45,omp45warn,omp %s

#include "Inputs/nesting_of_regions.cpp"
