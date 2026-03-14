// RUN: %clang_cc1 -triple  powerpc64le-unknown-unknown -std=c++11 -fopenmp -fexceptions -fcxx-exceptions -O0 -emit-llvm %s -o - | FileCheck %s

// Check that regions that install a terminate scope in the exception stack can
// correctly generate complex arithmetic.

// CHECK-LABEL: ffcomplex
void ffcomplex (int a) {
  double _Complex dc = (double)a;

  // CHECK: call noundef { double, double } @__muldc3(double noundef %{{.+}}, double noundef %{{.+}}, double noundef %{{.+}}, double noundef %{{.+}})
  dc *= dc;
  // CHECK: call {{.+}} @__kmpc_fork_call({{.+}} [[REGNAME1:@.*]], ptr %{{.+}})
  #pragma omp parallel
  {
    dc *= dc;
  }
  // CHECK: ret void
}

// CHECK: define internal {{.+}}[[REGNAME1]](
// CHECK-NOT: invoke
// CHECK: call noundef { double, double } @__muldc3(double noundef %{{.+}}, double noundef %{{.+}}, double noundef %{{.+}}, double noundef %{{.+}})
// CHECK-NOT: invoke
// CHECK: ret void

// Check if we are observing the function pointer attribute regardless what is
// in the exception specification of the callees.
void fnoexcp(void) noexcept;

// CHECK-LABEL: foo
void foo(int a, int b) {

  void (*fptr)(void) noexcept = fnoexcp;

  // CHECK: call {{.+}} @__kmpc_fork_call({{.+}} [[REGNAME2:@.*]], ptr %{{.+}})
  #pragma omp parallel
  {
    fptr();
  }
  // CHECK: ret void
}

// CHECK: define internal {{.+}}[[REGNAME2]](
// CHECK-NOT: invoke
// CHECK: call void %{{[0-9]+}}()
// CHECK-NOT: invoke
// CHECK: ret void
