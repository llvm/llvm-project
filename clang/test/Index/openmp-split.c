// RUN: c-index-test -test-load-source local %s -fopenmp=libomp -fopenmp-version=60 | FileCheck %s

void test(void) {
#pragma omp split counts(3, 7)
  for (int i = 0; i < 20; i += 1)
    ;
}

// CHECK: openmp-split.c:4:1: OMPSplitDirective= Extent=[4:1 - 4:31]
// CHECK: openmp-split.c:4:26: IntegerLiteral= Extent=[4:26 - 4:27]
// CHECK: openmp-split.c:4:29: IntegerLiteral= Extent=[4:29 - 4:30]
// CHECK: openmp-split.c:5:3: ForStmt= Extent=[5:3 - 6:6]
