// The decoupled drain reads only the canonical __llvm_profile_sections bounds
// table provided by the device profile runtime (InstrProfilingPlatformGPU.c),
// since clang no longer emits a per-TU struct. Assert that symbol is present
// in the device ELF's dynamic symbol table (protected visibility) for both
// non-RDC and RDC device compiles. This is the contract the drainer depends on.
//
// REQUIRES: hip, amdgpu

// RUN: rm -rf %t.dir && mkdir -p %t.dir

// --- non-RDC ---
// RUN: %clang -x hip --offload-arch=%amdgpu_arch -fno-gpu-rdc \
// RUN:   -fprofile-instr-generate -fcoverage-mapping %s -o %t.dir/a.out \
// RUN:   -L%hip_lib_path -lamdhip64
// Extraction failure here would make the readelf invocation succeed against
// an empty/missing file; surface it instead of hiding it behind `|| true`.
// RUN: llvm-objdump --offloading %t.dir/a.out > /dev/null
// RUN: llvm-readelf --dyn-syms %t.dir/a.out.0.hip-amdgcn-amd-amdhsa--*gfx* \
// RUN:   | FileCheck %s

// --- RDC ---
// RUN: %clang -x hip --offload-arch=%amdgpu_arch -fgpu-rdc \
// RUN:   -fprofile-instr-generate -fcoverage-mapping %s -o %t.dir/b.out \
// RUN:   -L%hip_lib_path -lamdhip64
// RUN: llvm-objdump --offloading %t.dir/b.out > /dev/null
// RUN: llvm-readelf --dyn-syms %t.dir/b.out.0.hip-amdgcn-amd-amdhsa--*gfx* \
// RUN:   | FileCheck %s

// CHECK: PROTECTED {{.*}} __llvm_profile_sections

#include <hip/hip_runtime.h>

__global__ void k(int *p) { *p += 1; }

int main() {
  int *d = nullptr;
  if (hipMalloc(&d, sizeof(int)) != hipSuccess)
    return 2;
  k<<<1, 1>>>(d);
  (void)hipFree(d);
  return 0;
}
