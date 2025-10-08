// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux -fopenmp-targets=spirv64-intel -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple spirv64-intel -fopenmp-targets=spirv64-intel -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - | FileCheck %s

// expected-no-diagnostics

// CHECK: call spir_func addrspace(9) void @__kmpc_parallel_51(ptr addrspace(4) addrspacecast (ptr addrspace(1) @{{.*}} to ptr addrspace(4)),
// CHECK-SAME: i32 %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, ptr addrspace(9) @{{.*}}, ptr addrspace(4) {{.*}}, ptr addrspace(4) %{{.*}}, i64 {{.*}})

// CHECK: call addrspace(9) i32 @__kmpc_nvptx_teams_reduce_nowait_v2(ptr addrspace(4) addrspacecast (ptr addrspace(1) @{{.*}} to ptr addrspace(4)),
// CHECK-SAME: ptr addrspace(4) %{{.*}}, i32 1024, i64 4, ptr addrspace(4) %{{.*}}, ptr addrspace(9) @{{.*}}, ptr addrspace(9) @{{.*}}, ptr addrspace(9) @{{.*}}, ptr addrspace(9) @{{.*}}, ptr addrspace(9) @{{.*}}, ptr addrspace(9) @{{.*}})

int main() {
  int matrix_sum = 0;
    #pragma omp target teams distribute parallel for \
                    reduction(+:matrix_sum) \
                    map(tofrom:matrix_sum)
    for (int i = 0; i < 100; i++) {

    }

    return 0;
}
