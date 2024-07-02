// RUN: %clang_cc1 -fsyntax-only -fopenmp -fopenmp-version=45 -fno-openmp-extensions -verify=expected,omp45,omp45warn,omp %s
// RUN: %clang_cc1 -fsyntax-only -fopenmp -fopenmp-version=45 -verify=expected,omp45,omp -fno-openmp-extensions -Wno-openmp %s

#include "Inputs/nesting_of_regions.cpp"
