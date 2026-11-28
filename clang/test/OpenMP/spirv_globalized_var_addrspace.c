// Globalized variables and the loop bounds that the parallel data-sharing
// wrapper reads from the shared variable list live in shared memory. On SPIR-V
// they must be accessed through generic (addrspace(4)) pointers: addrspace(0)
// is Function (private) storage there.

// RUN: %clang_cc1 -verify -fopenmp -x c -triple x86_64-unknown-linux -fopenmp-targets=spirv64-intel -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c -triple spirv64-intel -fopenmp-targets=spirv64-intel -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - | FileCheck %s

// expected-no-diagnostics

// CHECK-LABEL: define {{.*}}spir_kernel void @__omp_offloading_{{.*}}_globalized_local_l{{[0-9]+}}(
// CHECK:         [[X:%.*]] = call {{.*}}ptr addrspace(4) @__kmpc_alloc_shared(i64 4)
// CHECK-NOT:     addrspacecast ptr addrspace(4) [[X]] to ptr
// CHECK:         store i32 0, ptr addrspace(4) [[X]], align 4
int globalized_local(void) {
  int a = 0;
#pragma omp target map(from : a)
  {
    int x = 0;
#pragma omp parallel num_threads(4)
    x = 5;
    a = x;
  }
  return a;
}

// CHECK-LABEL: define internal spir_func void @__omp_offloading_{{.*}}_loop_bounds_l{{[0-9]+}}_omp_outlined_omp_outlined_wrapper(
// CHECK:         [[LIST:%.*]] = load ptr addrspace(4), ptr addrspace(4) %global_args.ascast
// CHECK-NEXT:    [[LBP:%.*]] = getelementptr inbounds ptr addrspace(4), ptr addrspace(4) [[LIST]], i64 0
// CHECK-NEXT:    [[LB:%.*]] = load i64, ptr addrspace(4) [[LBP]]
// CHECK-NEXT:    [[UBP:%.*]] = getelementptr inbounds ptr addrspace(4), ptr addrspace(4) [[LIST]], i64 1
// CHECK-NEXT:    [[UB:%.*]] = load i64, ptr addrspace(4) [[UBP]]
// CHECK:         call spir_func {{.*}}_loop_bounds_l{{[0-9]+}}_omp_outlined_omp_outlined(ptr addrspace(4) {{.*}}, ptr addrspace(4) {{.*}}, i64 [[LB]], i64 [[UB]],
void loop_bounds(int *a) {
#pragma omp target teams map(tofrom : a[:100])
  {
    a[0] = -1;
#pragma omp distribute parallel for
    for (int i = 1; i < 100; ++i)
      a[i] = i;
  }
}
