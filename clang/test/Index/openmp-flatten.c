// RUN: c-index-test -test-load-source local %s -fopenmp=libomp -fopenmp-version=61 | FileCheck %s

void test(void) {
#pragma omp flatten
  for (int i = 0; i < 20; i += 1)
    for (int j = 0; j < 30; j += 1)
      ;
}

// CHECK: openmp-flatten.c:4:1: OMPFlattenDirective= Extent=[4:1 - 4:20]
// CHECK: openmp-flatten.c:5:3: ForStmt= Extent=[5:3 - 7:8]
// CHECK: openmp-flatten.c:6:5: ForStmt= Extent=[6:5 - 7:8]
