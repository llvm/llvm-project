// no PCH
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fnoopenmp-use-tls -emit-llvm -include %s -include %s %s -o - | FileCheck %s
// with PCH
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fnoopenmp-use-tls -emit-llvm -chain-include %s -chain-include %s %s -o - | FileCheck %s
// no PCH
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -emit-llvm -include %s -include %s %s -o - | FileCheck %s -check-prefix=CHECK-TLS-1
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -emit-llvm -include %s -include %s %s -o - | FileCheck %s -check-prefix=CHECK-TLS-2
// with PCH
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -emit-llvm -chain-include %s -chain-include %s %s -o - | FileCheck %s -check-prefix=CHECK-TLS-1
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -emit-llvm -chain-include %s -chain-include %s %s -o - | FileCheck %s -check-prefix=CHECK-TLS-2

#if !defined(PASS1)
#define PASS1

extern "C" int* malloc (int size);
int *a = malloc(20);

#elif !defined(PASS2)
#define PASS2

#pragma omp threadprivate(a)

#else

// CHECK: call {{.*}} @__kmpc_threadprivate_register(
// CHECK-TLS-1: @{{a|\"\?a@@3PE?AHE?A\"}} = {{.*}}thread_local {{.*}}global {{.*}}i32*

// CHECK-LABEL: foo
// CHECK-TLS-LABEL: foo
int foo() {
  return *a;
  // CHECK: call {{.*}} @__kmpc_global_thread_num(
  // CHECK: call {{.*}} @__kmpc_threadprivate_cached(
  // CHECK-TLS-1: call {{.*}} @{{_ZTW1a|\"\?\?__Ea@@YAXXZ\"}}()
}

// CHECK-TLS-2: define {{.*}} @{{_ZTW1a|\"\?\?__Ea@@YAXXZ\"}}()

#endif
