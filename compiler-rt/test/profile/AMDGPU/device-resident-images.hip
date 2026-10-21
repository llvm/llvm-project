// A host-registered HIP image with no launched kernel must stay unloaded when
// profiles are collected at exit. Build two non-RDC translation units so each
// has its own static device image. Only the hot image executes.
//
// REQUIRES: hip, amdgpu
// UNSUPPORTED: windows
// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: %clang -x hip --offload-arch=%amdgpu_arch -fno-gpu-rdc \
// RUN:   -fprofile-instr-generate -c %s -o %t.dir/hot.o
// RUN: %clang -x hip --offload-arch=%amdgpu_arch -fno-gpu-rdc \
// RUN:   -fprofile-instr-generate -DCOLD_IMAGE -c %s -o %t.dir/cold.o
// RUN: %clang --hip-link --offload-arch=%amdgpu_arch \
// RUN:   -fprofile-instr-generate %t.dir/hot.o %t.dir/cold.o \
// RUN:   -o %t.dir/a.out -L%hip_lib_path -lamdhip64
// RUN: cd %t.dir && llvm-objdump --offloading a.out \
// RUN:   | FileCheck %s --check-prefix=IMAGES
// RUN: env LLVM_PROFILE_FILE=%t.dir/host.%%p.profraw \
// RUN:   LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH \
// RUN:   HIP_VISIBLE_DEVICES=0 %t.dir/a.out
// RUN: /bin/sh -c 'test "$(find %t.dir -name "gfx*.profraw" | wc -l)" -eq 1'
// RUN: llvm-profdata show --all-functions --counts %t.dir/gfx*.profraw \
// RUN:   | FileCheck %s --check-prefix=PROFILE --implicit-check-not=cold_kernel

#include <hip/hip_runtime.h>

#ifdef COLD_IMAGE

extern "C" __global__ void cold_kernel(int *P) { ++*P; }
extern "C" void cold_host_anchor() {}

#else

extern "C" void cold_host_anchor();
extern "C" __global__ void hot_kernel(int *P) { ++*P; }

int main() {
  cold_host_anchor(); // Keep the cold translation unit linked into the host.
  int *Device = nullptr;
  if (hipMalloc(&Device, sizeof(int)) != hipSuccess)
    return 1;
  int Value = 0;
  if (hipMemcpy(Device, &Value, sizeof(Value), hipMemcpyHostToDevice) !=
      hipSuccess)
    return 2;
  hot_kernel<<<1, 1>>>(Device);
  if (hipMemcpy(&Value, Device, sizeof(Value), hipMemcpyDeviceToHost) !=
      hipSuccess)
    return 3;
  if (hipFree(Device) != hipSuccess)
    return 4;
  return Value == 1 ? 0 : 5;
}

#endif

// IMAGES-DAG: a.out.0.hip-amdgcn-amd-amdhsa--
// IMAGES-DAG: a.out.1.hip-amdgcn-amd-amdhsa--
// PROFILE: hot_kernel:
// PROFILE: Function count: 1
