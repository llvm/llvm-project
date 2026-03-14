// RUN: c-index-test -test-load-source local %s -fopenmp=libomp | FileCheck %s

void test() {
#pragma omp tile sizes(5)
  for (int i = 0; i < 65; i += 1)
    ;
}

// CHECK: openmp-tile.c:4:1: OMPTileDirective= Extent=[4:1 - 4:26]
// CHECK: openmp-tile.c:4:24: IntegerLiteral= Extent=[4:24 - 4:25]
// CHECK: openmp-tile.c:5:3: ForStmt= Extent=[5:3 - 6:6]
