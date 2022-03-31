// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=99 -DOMP99 -ast-print %s | FileCheck --check-prefixes=CHECK,REV %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=99 -DOMP99 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=99 -DOMP99 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck --check-prefixes=CHECK,REV %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=99 -DOMP99 -ast-print %s | FileCheck --check-prefixes=CHECK,REV %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=99 -DOMP99 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=99 -DOMP99 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck --check-prefixes=CHECK,REV %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

#pragma omp requires unified_address 
// CHECK:#pragma omp requires unified_address

#pragma omp requires unified_shared_memory
// CHECK:#pragma omp requires unified_shared_memory

#ifdef OMP99
#pragma omp requires reverse_offload
// REV:#pragma omp requires reverse_offload
#endif

#pragma omp requires dynamic_allocators
// CHECK:#pragma omp requires dynamic_allocators

#pragma omp requires atomic_default_mem_order(seq_cst)
// CHECK:#pragma omp requires atomic_default_mem_order(seq_cst)

#endif
