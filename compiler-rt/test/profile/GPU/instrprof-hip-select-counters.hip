// NOTE: Do not autogenerate. Check runtime counts and application output.
// Select increments vary across active lanes. Check the serialized device
// counts as well as application output; both select arms contribute equally.
// REQUIRES: hip, amdgpu
// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: %clang -x hip --offload-arch=%amdgpu_arch -O2 -fprofile-generate \
// RUN:   -mllvm -pgo-instrument-entry %s -o %t.dir/a.out \
// RUN:   -L%hip_lib_path -lamdhip64
// RUN: env LLVM_PROFILE_FILE=%t.dir/profile.%%p.profraw \
// RUN:   LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH \
// RUN:   HIP_VISIBLE_DEVICES=0 %run %t.dir/a.out
// RUN: llvm-profdata merge %t.dir/%amdgpu_arch*.profraw -o %t.dir/a.profdata
// RUN: llvm-profdata show --all-functions --counts %t.dir/a.profdata | FileCheck %s
// CHECK-LABEL: {{^  }}select_values:
// CHECK: Counters: 2
// CHECK: Block counts: [64, 32]

#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

extern "C" __global__ void select_values(volatile float *input, float *output) {
  unsigned i = threadIdx.x;
  float a = input[i];
  float b = input[i + 64];
  output[i] = (i & 1) ? a : b;
}

static void check(hipError_t e) {
  if (e != hipSuccess)
    std::exit(2);
}

int main() {
  float input[128], output[64], *device_input, *device_output;
  for (unsigned i = 0; i < 128; ++i)
    input[i] = i + 1;
  check(hipMalloc(&device_input, sizeof(input)));
  check(hipMalloc(&device_output, sizeof(output)));
  check(hipMemcpy(device_input, input, sizeof(input), hipMemcpyHostToDevice));
  select_values<<<1, 64>>>(device_input, device_output);
  check(hipGetLastError());
  check(
      hipMemcpy(output, device_output, sizeof(output), hipMemcpyDeviceToHost));
  for (unsigned i = 0; i < 64; ++i)
    if (output[i] != input[(i & 1) ? i : i + 64])
      return 1;
  check(hipFree(device_output));
  check(hipFree(device_input));
  std::puts("select_values output correct; expected entry=64 true=32 false=32");
}
