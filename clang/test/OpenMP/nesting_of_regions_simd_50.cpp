// RUN: %clang_cc1 -fsyntax-only -fopenmp-simd -verify=expected,omp50,omp -fno-openmp-extensions %s

#include "Inputs/nesting_of_regions.cpp"
