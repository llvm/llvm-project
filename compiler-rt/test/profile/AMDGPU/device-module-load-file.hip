// Device PGO supports code objects loaded by filename through hipModuleLoad.
//
// REQUIRES: hip, amdgpu
// UNSUPPORTED: windows

// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: %clang -x hip --offload-arch=%amdgpu_arch \
// RUN:   --rocm-path=%hip_lib_path/.. \
// RUN:   -fprofile-instr-generate -DBUILD_MODULE %s \
// RUN:   -o %t.dir/module -L%hip_lib_path -lamdhip64
// RUN: cd %t.dir && llvm-objdump --offloading module
// RUN: mv %t.dir/module.0.hip-amdgcn-amd-amdhsa--*gfx* %t.dir/module.co
// RUN: %clang -x c++ -c %s -D__HIP_PLATFORM_AMD__ \
// RUN:   -I%hip_lib_path/../include -o %t.dir/loader.o
// RUN: %clang %t.dir/loader.o --hip-link \
// RUN:   --rocm-path=%hip_lib_path/.. -fprofile-instr-generate \
// RUN:   -L%hip_lib_path -lamdhip64 -o %t.dir/loader
// RUN: env LLVM_PROFILE_FILE=%t.dir/host.%%p.profraw \
// RUN:   LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH \
// RUN:   HIP_VISIBLE_DEVICES=0 %t.dir/loader %t.dir/module.co
// RUN: ls %t.dir/gfx*.profraw
// RUN: llvm-profdata merge %t.dir/*.profraw -o %t.dir/module.profdata
// RUN: llvm-profdata show --all-functions %t.dir/module.profdata \
// RUN:   | FileCheck %s

#include <hip/hip_runtime.h>

#ifdef BUILD_MODULE

extern "C" __global__ void module_kernel() {}

int main() { return 0; }

#else

int main(int Argc, char **Argv) {
  if (Argc != 2)
    return 1;

  hipModule_t Module = nullptr;
  if (hipModuleLoad(&Module, Argv[1]) != hipSuccess)
    return 2;

  hipFunction_t Function = nullptr;
  if (hipModuleGetFunction(&Function, Module, "module_kernel") != hipSuccess)
    return 3;

  if (hipModuleLaunchKernel(Function, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr,
                            nullptr) != hipSuccess)
    return 4;
  if (hipDeviceSynchronize() != hipSuccess)
    return 5;
  if (hipModuleUnload(Module) != hipSuccess)
    return 6;
  return 0;
}

#endif

// CHECK:      module_kernel:
// CHECK-NEXT:   Hash:
// CHECK-NEXT:   Counters: 1
// CHECK-NEXT:   Function count: 1
