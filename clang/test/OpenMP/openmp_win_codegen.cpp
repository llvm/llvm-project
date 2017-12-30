// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-pc-windows-msvc18.0.0 -std=c++11 -fms-compatibility-version=18 -fms-extensions -emit-llvm %s -fexceptions -fcxx-exceptions -o - -O1 | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple x86_64-pc-windows-msvc18.0.0 -std=c++11 -fms-compatibility-version=18 -fms-extensions -emit-llvm %s -fexceptions -fcxx-exceptions -o - -O1 | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// REQUIRES: x86-registered-target
// expected-no-diagnostics

void foo();
void bar();

struct Test {
  static void main() {
    int failed = 0;
    int j = 2;

#pragma omp parallel
    {
      int local_j = 3;
#pragma omp single copyprivate(local_j)
      {
        local_j = 4;
      }

      // Assure reports a data race, but value written to "j"
      // should always be the same.
      j = local_j;
    }

  }
};

// CHECK-LABEL: @main
int main() {
  // CHECK: call void @{{.+}}main
  Test::main();
  // CHECK: call void (%ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%ident_t* {{.*}}@0, i32 0, void (i32*, i32*, ...)* bitcast (void (i32*, i32*)* [[OUTLINED:@.+]] to void (i32*, i32*, ...)*))
#pragma omp parallel
  {
    try {
      foo();
    } catch (int t) {
#pragma omp critical
      {
        bar();
      };
    }
  };
  // CHECK: ret i32 0
  return 0;
}

// CHECK: define internal void [[OUTLINED]](
// CHECK: [[GID:%.+]] = {{.*}}call i32 @__kmpc_global_thread_num(%ident_t* {{.*}}@0)
// CHECK: invoke void @{{.+}}foo
// CHECK: catchswitch within
// CHECK: catchpad within
// CHECK: call void @__kmpc_critical(%ident_t* {{.*}}@0, i32 [[GID]],
// CHECK: invoke void @{{.+}}bar
// CHECK: call void @__kmpc_end_critical(%ident_t* {{.*}}@0, i32 [[GID]],
// CHECK: catchret from
// CHECK: cleanuppad within
// CHECK: call void @__kmpc_end_critical(%ident_t* {{.*}}@0, i32 [[GID]],
// CHECK: cleanupret from

