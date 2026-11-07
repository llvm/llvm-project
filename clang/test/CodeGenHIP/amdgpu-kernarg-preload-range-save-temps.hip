// REQUIRES: amdgpu-registered-target

// RUN: rm -rf %t && mkdir -p %t
// RUN: cd %t && %clang --target=x86_64-unknown-linux-gnu --offload-arch=gfx942 \
// RUN:   --cuda-device-only -nogpulib -nogpuinc -save-temps -O2 -c -x hip %s \
// RUN:   -o out.o
// RUN: FileCheck %s --input-file=%t/amdgpu-kernarg-preload-range-save-temps-hip-amdgcn-amd-amdhsa-gfx942.s
// RUN: %clang --target=x86_64-unknown-linux-gnu --offload-arch=gfx942 \
// RUN:   --cuda-device-only -nogpulib -nogpuinc -O0 -S -x hip %s \
// RUN:   -o %t/unoptimized.s
// RUN: FileCheck %s --check-prefix=O0 --input-file=%t/unoptimized.s
// RUN: not %clang --target=x86_64-unknown-linux-gnu --offload-arch=gfx942 \
// RUN:   --cuda-device-only -nogpulib -nogpuinc -O2 -S -x hip \
// RUN:   -DUNSUPPORTED_AGGREGATE %s -o %t/unsupported.s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ERROR

extern "C" __attribute__((global, amdgpu_kernarg_preload(1, 2)))
void preload_range_1_2(int *out, int a, int b) {
  *out = a + b;
}

extern "C" __attribute__((global, amdgpu_kernarg_preload(0, 0)))
void preload_range_0_0(int *out, int a, int b) {
  *out = a + b;
}

#ifdef UNSUPPORTED_AGGREGATE
struct UnsupportedAggregate {
  int value[2];
};

extern "C" __attribute__((global, amdgpu_kernarg_preload(0, 0)))
void preload_unsupported_aggregate(UnsupportedAggregate arg) {}
#endif

// CHECK-LABEL: .amdhsa_kernel preload_range_1_2
// CHECK: .amdhsa_user_sgpr_kernarg_preload_length 2
// CHECK: .amdhsa_user_sgpr_kernarg_preload_offset 2

// CHECK-LABEL: .amdhsa_kernel preload_range_0_0
// CHECK: .amdhsa_user_sgpr_kernarg_preload_length 2
// CHECK: .amdhsa_user_sgpr_kernarg_preload_offset 0

// O0-LABEL: .amdhsa_kernel preload_range_1_2
// O0: .amdhsa_user_sgpr_kernarg_preload_length 0
// O0: .amdhsa_user_sgpr_kernarg_preload_offset 0

// O0-LABEL: .amdhsa_kernel preload_range_0_0
// O0: .amdhsa_user_sgpr_kernarg_preload_length 0
// O0: .amdhsa_user_sgpr_kernarg_preload_offset 0

// ERROR: error: unsupported argument in amdgpu kernarg preload range
