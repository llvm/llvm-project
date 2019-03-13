// no PCH
// RUN: %clang_cc1 -fopenmp -fnoopenmp-use-tls -ast-print -include %s -include %s %s -o - | FileCheck %s
// with PCH
// RUN: %clang_cc1 -fopenmp -fnoopenmp-use-tls -ast-print -chain-include %s -chain-include %s %s -o - | FileCheck %s
// no PCH
// RUN: %clang_cc1 -fopenmp -ast-print -include %s -include %s %s -o - | FileCheck %s -check-prefix=CHECK-ALLOC-1
// RUN: %clang_cc1 -fopenmp -ast-print -include %s -include %s %s -o - | FileCheck %s -check-prefix=CHECK-ALLOC-2
// with PCH
// RUN: %clang_cc1 -fopenmp -ast-print -chain-include %s -chain-include %s %s -o - | FileCheck %s -check-prefix=CHECK-ALLOC-1
// RUN: %clang_cc1 -fopenmp -ast-print -chain-include %s -chain-include %s %s -o - | FileCheck %s -check-prefix=CHECK-ALLOC-2

#if !defined(PASS1)
#define PASS1

typedef void **omp_allocator_handle_t;
extern const omp_allocator_handle_t omp_default_mem_alloc;

int a;
// CHECK: int a;

#elif !defined(PASS2)
#define PASS2

#pragma omp allocate(a) allocator(omp_default_mem_alloc)
// CHECK: #pragma omp allocate(a) allocator(omp_default_mem_alloc)

#else

// CHECK-LABEL: foo
// CHECK-ALLOC-LABEL: foo
int foo() {
  return a;
  // CHECK: return a;
  // CHECK-ALLOC-1: return a;
}

// CHECK-ALLOC-2: return a;

#endif
