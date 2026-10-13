// Independence / robustness: an instrumented HIP program that never launches a
// kernel still writes its host .profraw, and the device drain is a clean no-op
// (no crash, no spurious device .profraw). We assert the no-op condition
// directly via the runtime's verbose log rather than rely on HIP lazy-loading
// to leave the device code object unloaded -- the loader may load it for
// other reasons (e.g. eager registration), and in that case the drain
// legitimately walks it and reports zero instrumented sections / zero
// drained. Either outcome is correct.
//
// REQUIRES: hip, amdgpu
// The terminal conditions checked below ("no GPU agents", "no loaded
// segments", "drained=0") are Linux HSA-drain strings with no Windows analog.
// UNSUPPORTED: windows

// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: %clang -x hip --offload-arch=%amdgpu_arch \
// RUN:   -fprofile-instr-generate -fcoverage-mapping %s -o %t.dir/a.out \
// RUN:   -L%hip_lib_path -lamdhip64
// RUN: env LLVM_PROFILE_FILE=%t.dir/host.%%p.profraw \
// RUN:   LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH \
// RUN:   LLVM_PROFILE_VERBOSE=1 \
// RUN:   %t.dir/a.out 2> %t.dir/verbose.log
// RUN: ls %t.dir/host.*.profraw
// No arch-prefixed device profraw should have been produced.
// RUN: not ls %t.dir/gfx*.profraw
// The drain must have run; one of these three terminal conditions must hold:
//   - no GPU agents enumerated (test host has /dev/kfd but no usable agent)
//   - no loaded code object segments at exit
//   - the walk completed and drained=0 (no instrumented kernel was launched
//     so the device code object either wasn't loaded or its bounds were
//     empty/already drained)
// RUN: FileCheck --input-file=%t.dir/verbose.log %s
// CHECK: {{no GPU agents present|no loaded segments|drained=0}}

#include <hip/hip_runtime.h>

// Defined but never launched.
__global__ void unused(int *p) { *p += 1; }

int main() {
  int n = 0;
  (void)hipGetDeviceCount(&n);
  return 0;
}
