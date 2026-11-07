// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 -x c++ \
// RUN:   -fopenmp-metadirective-runtime-selection \
// RUN:   -triple x86_64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:   -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 -x c++ \
// RUN:   -fopenmp-metadirective-runtime-selection \
// RUN:   -triple nvptx64-nvidia-cuda -aux-triple x86_64-unknown-unknown \
// RUN:   -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s \
// RUN:   -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc \
// RUN:   -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 -x c++ \
// RUN:   -fopenmp-metadirective-runtime-selection \
// RUN:   -triple x86_64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=HOST
// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 -x c++ \
// RUN:   -fopenmp-metadirective-runtime-selection \
// RUN:   -triple nvptx64-nvidia-cuda -aux-triple x86_64-unknown-unknown \
// RUN:   -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s \
// RUN:   -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc \
// RUN:   -o - | FileCheck %s --check-prefix=DEV
// expected-no-diagnostics

// A target region in a variant of a runtime metadirective gets a device
// kernel.
// CHECK: define {{.*}}void @__omp_offloading_{{.*}}_Z14target_varianti_l{{[0-9]+}}(
void target_variant(int flag) {
  int x = 0;
#pragma omp metadirective when(user={condition(flag)}: target map(tofrom: x)) otherwise(parallel)
  {
    x = 1;
  }
}

// Two target variants on one line: their kernels differ only by a counter, so
// host and device must number them the same way. The host launches the
// condition(flag) variant (thread_limit 9) as _1, and the device kernel _1 is
// that variant (9 plus the 64 threads added for generic mode).
// HOST: call i32 @__tgt_target_kernel({{.*}}, i32 9, ptr @.__omp_offloading_{{.*}}_Z9same_linei_l{{[0-9]+}}_1.region_id
// DEV-DAG: @__omp_offloading_{{.*}}_Z9same_linei_l{{[0-9]+}}_kernel_environment = {{.*}}, i32 69, i32 -1
// DEV-DAG: @__omp_offloading_{{.*}}_Z9same_linei_l{{[0-9]+}}_1_kernel_environment = {{.*}}, i32 73, i32 -1
void same_line(int flag) {
  int x = 0;
#pragma omp metadirective when(implementation={vendor(amd)}: target thread_limit(5) map(tofrom: x)) when(user={condition(flag)}: target thread_limit(9) map(tofrom: x))
  x = 1;
}

// On separate lines, each variant's kernel has its own line number.
// HOST: call i32 @__tgt_target_kernel({{.*}}, i32 9, ptr @.__omp_offloading_{{.*}}_Z14separate_linesi_l{{[0-9]+}}.region_id
// DEV-DAG: @__omp_offloading_{{.*}}_Z14separate_linesi_l{{[0-9]+}}_kernel_environment = {{.*}}, i32 69, i32 -1
// DEV-DAG: @__omp_offloading_{{.*}}_Z14separate_linesi_l{{[0-9]+}}_kernel_environment = {{.*}}, i32 73, i32 -1
void separate_lines(int flag) {
  int x = 0;
#pragma omp metadirective \
    when(implementation={vendor(amd)}: target thread_limit(5) map(tofrom: x)) \
    when(user={condition(flag)}: target thread_limit(9) map(tofrom: x))
  x = 1;
}
