// Basic HIP device PGO drain end-to-end: a host + device .profraw are written
// at exit (the device one arch-prefixed), they merge, the merged profile
// contains the device kernel's counters, and llvm-cov reports device-side
// coverage. Covers both non-RDC and RDC device compiles.
//
// REQUIRES: hip, amdgpu

// RUN: rm -rf %t.dir && mkdir -p %t.dir

// --- non-RDC ---
// RUN: %clang -x hip --offload-arch=%amdgpu_arch -fno-gpu-rdc \
// RUN:   -fprofile-instr-generate -fcoverage-mapping %s -o %t.dir/a.out \
// RUN:   -L%hip_lib_path -lamdhip64
// RUN: env LLVM_PROFILE_FILE=%t.dir/host.%%p.profraw \
// RUN:   LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH \
// RUN:   %t.dir/a.out
// A device profraw (arch-prefixed) must have been drained alongside the host one.
// RUN: ls %t.dir/gfx*.profraw
// RUN: llvm-profdata merge %t.dir/*.profraw -o %t.dir/a.profdata
// RUN: llvm-profdata show --all-functions %t.dir/a.profdata \
// RUN:   | FileCheck --check-prefix=FUNCS %s
// Confirm the embedded device image is extractable (failure here is the real
// cause of any downstream llvm-cov failure, so let it propagate).
// RUN: llvm-objdump --offloading %t.dir/a.out > /dev/null
// RUN: llvm-cov report %t.dir/a.out.0.hip-amdgcn-amd-amdhsa--*gfx* \
// RUN:   -instr-profile=%t.dir/a.profdata 2>&1 | FileCheck --check-prefix=COV %s

// --- RDC ---
// RUN: rm -f %t.dir/*.profraw
// RUN: %clang -x hip --offload-arch=%amdgpu_arch -fgpu-rdc \
// RUN:   -fprofile-instr-generate -fcoverage-mapping %s -o %t.dir/b.out \
// RUN:   -L%hip_lib_path -lamdhip64
// RUN: env LLVM_PROFILE_FILE=%t.dir/host.%%p.profraw \
// RUN:   LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH \
// RUN:   %t.dir/b.out
// RUN: ls %t.dir/gfx*.profraw
// RUN: llvm-profdata merge %t.dir/*.profraw -o %t.dir/b.profdata
// RUN: llvm-profdata show --all-functions %t.dir/b.profdata \
// RUN:   | FileCheck --check-prefix=FUNCS %s

#include <hip/hip_runtime.h>

__global__ void addk(int *p) {
  if (*p > 0)
    *p += 1;
  else
    *p -= 1;
}

int main() {
  int *d = nullptr;
  if (hipMalloc(&d, sizeof(int)) != hipSuccess)
    return 2;
  int h = 5;
  (void)hipMemcpy(d, &h, sizeof(int), hipMemcpyHostToDevice);
  addk<<<1, 1>>>(d);
  (void)hipMemcpy(&h, d, sizeof(int), hipMemcpyDeviceToHost);
  (void)hipFree(d);
  return h > 0 ? 0 : 1;
}

// The merged profile contains both the host main and the device kernel,
// proving the device counters were drained and merged.
// FUNCS-DAG: addk
// FUNCS-DAG: main

// COV: TOTAL
