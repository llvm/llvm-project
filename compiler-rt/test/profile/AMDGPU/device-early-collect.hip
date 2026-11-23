// M1 regression: calling __llvm_profile_hip_collect_device_data() before any
// kernel has been launched must not poison the later atexit drain.  The early
// call sees "no instrumented code object loaded yet" (a transient no-op) and
// must not latch the drain as completed; otherwise the post-launch atexit
// pass produces no device .profraw and we silently lose device counters.
//
// REQUIRES: hip, amdgpu
// Guards the Linux introspection drain's DrainCompleted latch; the Windows
// host-shadow drain has no such latch (it tracks per-TU Processed flags).
// UNSUPPORTED: windows

// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: %clang -x hip --offload-arch=%amdgpu_arch -fno-gpu-rdc \
// RUN:   -fprofile-instr-generate -fcoverage-mapping %s -o %t.dir/a.out \
// RUN:   -L%hip_lib_path -lamdhip64
// RUN: env LLVM_PROFILE_FILE=%t.dir/host.%%p.profraw \
// RUN:   LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH \
// RUN:   %t.dir/a.out
// Both the host profraw and at least one device profraw (gfx-prefixed) must
// have been produced, despite the early collection attempt.
// RUN: ls %t.dir/host.*.profraw
// RUN: ls %t.dir/gfx*.profraw
// And the merged profile must contain the device kernel that was launched
// *after* the early collect.
// RUN: llvm-profdata merge %t.dir/*.profraw -o %t.dir/a.profdata
// RUN: llvm-profdata show --all-functions %t.dir/a.profdata \
// RUN:   | FileCheck %s

#include <hip/hip_runtime.h>

// Declared by libclang_rt.profile-<host arch>.a; we call it directly to
// simulate any caller that drains device counters at an arbitrary point in
// the program lifetime (e.g. a per-iteration profile dump).
extern "C" int __llvm_profile_hip_collect_device_data(void);

__global__ void post_collect_kernel(int *p) {
  if (*p > 0)
    *p += 1;
  else
    *p -= 1;
}

int main() {
  // (1) Early collection -- runs before any kernel launch.  The drainer
  //     finds either no GPU agents, no loaded segments, or no instrumented
  //     bounds table, and returns 0 without latching DrainCompleted.
  (void)__llvm_profile_hip_collect_device_data();

  // (2) Now launch a kernel.  HIP loads the device code object that carries
  //     the __llvm_profile_sections bounds table, executes our kernel, and
  //     populates the device-side counters.
  int *d = nullptr;
  if (hipMalloc(&d, sizeof(int)) != hipSuccess)
    return 2;
  int h = 5;
  (void)hipMemcpy(d, &h, sizeof(int), hipMemcpyHostToDevice);
  post_collect_kernel<<<1, 1>>>(d);
  (void)hipMemcpy(&h, d, sizeof(int), hipMemcpyDeviceToHost);
  (void)hipFree(d);

  // (3) Exit normally.  The atexit drain runs and -- because step (1) did
  //     not latch DrainCompleted -- it walks the (now loaded) code object,
  //     finds __llvm_profile_sections, and emits the device .profraw.
  return h > 0 ? 0 : 1;
}

// CHECK-DAG: post_collect_kernel
// CHECK-DAG: main
