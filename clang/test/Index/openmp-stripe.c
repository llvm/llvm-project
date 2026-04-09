// RUN: c-index-test -test-load-source local %s -fopenmp=libomp -fopenmp-version=60 | FileCheck %s

void test() {
#pragma omp stripe sizes(5)
  for (int i = 0; i < 65; i += 1)
    ;
}

// CHECK: openmp-stripe.c:4:1: OMPStripeDirective= Extent=[4:1 - 4:28]
// CHECK: openmp-stripe.c:4:26: IntegerLiteral= Extent=[4:26 - 4:27]
// CHECK: openmp-stripe.c:5:3: ForStmt= Extent=[5:3 - 6:6]
