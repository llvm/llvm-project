// RUN: %clang_cc1 -verify -fopenmp -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -triple x86_64-unknown-linux-gnu -fopenmp-targets=x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,OFFLOAD
// expected-no-diagnostics

// The outlined target region takes a variable from a non-default address space
// as a plain pointer, so the host fallback call has to cast its address (GH140069).

int __seg_gs a;
int b;

// CHECK-DAG: @a = {{.*}}addrspace(256) global i32 0
// CHECK-DAG: @b = {{.*}}global i32 0

// CHECK-LABEL: define {{.*}}void @f(
// OFFLOAD:     call i32 @__tgt_target_kernel(
// CHECK:       call void @[[OUTLINED:__omp_offloading_[0-9a-z]+_[0-9a-z]+_f_l[0-9]+]](ptr addrspacecast (ptr addrspace(256) @a to ptr), ptr null)
void f(void) {
#pragma omp target map(alloc: a) map(from: b)
  {
    a = 0;
  }
}

// CHECK:       define internal void @[[OUTLINED]](ptr noundef nonnull align 4 dereferenceable(4) %{{.+}}, ptr noalias noundef %{{.+}})
// CHECK-NOT:   addrspace(256)
// CHECK:       store i32 0, ptr %{{.+}}, align 4

// CHECK-LABEL: define {{.*}}void @g(
// OFFLOAD:     call i32 @__tgt_target_kernel(
// CHECK:       call void @[[OUTLINED2:__omp_offloading_[0-9a-z]+_[0-9a-z]+_g_l[0-9]+]](ptr addrspacecast (ptr addrspace(256) @a to ptr), ptr @b, ptr null)
void g(void) {
#pragma omp target map(alloc: a) map(from: b)
  {
    a = 321;
    b = a;
  }
}

// CHECK:       define internal void @[[OUTLINED2]](ptr noundef nonnull align 4 dereferenceable(4) %{{.+}}, ptr noundef nonnull align 4 dereferenceable(4) %{{.+}}, ptr noalias noundef %{{.+}})
// CHECK-NOT:   addrspace(256)
// CHECK:       store i32 321, ptr %{{.+}}, align 4
