// RUN: %clang_cc1 -fsyntax-only -fopenmp -fno-openmp-extensions -verify=expected,omp50,omp %s
// RUN: %clang_cc1 -fsyntax-only -fopenmp -fopenmp-extensions -verify=expected,omp50 %s

#include "Inputs/nesting_of_regions.cpp"
