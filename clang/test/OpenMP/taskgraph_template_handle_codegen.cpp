// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

template <typename T>
void templ_handle(T &x) {
#pragma omp taskgraph
  {
#pragma omp task shared(x)
    {
      x += 1;
    }
  }
}

int main() {
  int i = 0;
  long l = 0;
  templ_handle(i);
  templ_handle(l);
  return 0;
}

// CHECK-DAG: @[[H1:.omp.taskgraph.handle[^ ]*]] = internal global ptr null
// CHECK-DAG: @[[H2:.omp.taskgraph.handle[^ ]*]] = internal global ptr null

// CHECK-LABEL: define linkonce_odr {{.*}} @_Z12templ_handleIiEvRT_(
// CHECK: call void @__kmpc_taskgraph(ptr {{[^,]+}}, i32 {{[^,]+}}, ptr @[[H1]], i64 0, i32 0, i32 0, ptr {{[^,]+}}, ptr {{[^)]+}})

// CHECK-LABEL: define linkonce_odr {{.*}} @_Z12templ_handleIlEvRT_(
// CHECK-NOT: ptr @[[H1]]
// CHECK: call void @__kmpc_taskgraph(ptr {{[^,]+}}, i32 {{[^,]+}}, ptr @[[H2]], i64 0, i32 0, i32 0, ptr {{[^,]+}}, ptr {{[^)]+}})

#endif
