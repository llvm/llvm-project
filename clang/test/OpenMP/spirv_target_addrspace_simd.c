// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-unknown -fopenmp-targets=spirv64 -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=spirv64 -fopenmp-is-target-device -triple spirv64 -fopenmp-host-ir-file-path %t-host.bc -emit-llvm %s -o - | FileCheck %s

int main() {
  int x = 0;

#pragma omp target teams distribute parallel for simd
  for(int i = 0; i < 1024; i++)
    x+=i;
  return x;
}

// CHECK: @[[#STRLOC:]] = private unnamed_addr addrspace(1) constant [{{.*}} x i8] c{{.*}}, align 1
// CHECK: @[[#IDENT:]] = private unnamed_addr addrspace(1) constant %struct.ident_t { i32 {{.*}}, i32 2050, i32 {{.*}}, i32 {{.*}}, ptr addrspacecast (ptr addrspace(1) @[[#STRLOC]] to ptr) }, align 8
// CHECK: define internal spir_func void @__omp_offloading_{{.*}}_omp_outlined(ptr addrspace(4) noalias noundef {{.*}}., ptr addrspace(4) noalias noundef {{.*}}, i64 noundef {{.*}}) #{{.*}} {
// CHECK: = load ptr addrspace(4), ptr addrspace(4) %{{.*}}, align 8
// CHECK: = load i32, ptr addrspace(4) %{{.*}}, align 4
// CHECK: = addrspacecast ptr addrspace(4) %{{.*}} to ptr
// CHECK: = addrspacecast ptr addrspace(4) %{{.*}} to ptr
// CHECK: = addrspacecast ptr addrspace(4) %{{.*}} to ptr
// CHECK: = addrspacecast ptr addrspace(4) %{{.*}} to ptr
// CHECK: call spir_func void @__kmpc_distribute_static_init{{.*}}(ptr addrspacecast (ptr addrspace(1) @[[#IDENT]] to ptr), i32 %{{.*}}, i32 {{.*}}, ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}}, i32 {{.*}}, i32 %{{.*}})
// CHECK: call spir_func void @__kmpc_distribute_static_fini{{.*}}(ptr addrspacecast (ptr addrspace(1) @[[#IDENT]] to ptr), i32 %{{.*}})
