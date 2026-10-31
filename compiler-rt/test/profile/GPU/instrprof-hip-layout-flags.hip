// NOTE: Do not autogenerate. Check runtime profile flags and reconstructed entry count.
// Device profile layout flags must be independent of the host flags.
// REQUIRES: hip, amdgpu
//
// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: mkdir %t.dir/entry
// RUN: %clang -x hip -O2 -Xarch_device -nogpulib -Xarch_device -nogpuinc \
// RUN:   -fprofile-generate --offload-arch=%amdgpu_arch \
// RUN:   --rocm-path=%hip_lib_path/.. %s -L%hip_lib_path -lamdhip64 \
// RUN:   -Xarch_device -mllvm=-pgo-instrument-entry \
// RUN:   -o %t.dir/entry/test
// RUN: env LLVM_PROFILE_FILE=%t.dir/entry/host.profraw \
// RUN:   LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH \
// RUN:   HIP_VISIBLE_DEVICES=0 %run %t.dir/entry/test
// RUN: llvm-profdata show %t.dir/entry/gfx*.profraw | FileCheck %s --check-prefix=ENTRY
// RUN: llvm-profdata show %t.dir/entry/host.profraw | FileCheck %s --check-prefix=NEITHER
// RUN: mkdir %t.dir/loop
// RUN: %clang -x hip -O2 -Xarch_device -nogpulib -Xarch_device -nogpuinc \
// RUN:   -fprofile-generate --offload-arch=%amdgpu_arch \
// RUN:   --rocm-path=%hip_lib_path/.. %s -L%hip_lib_path -lamdhip64 \
// RUN:   -Xarch_device -mllvm=-pgo-instrument-loop-entries \
// RUN:   -o %t.dir/loop/test
// RUN: env LLVM_PROFILE_FILE=%t.dir/loop/host.profraw \
// RUN:   LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH \
// RUN:   HIP_VISIBLE_DEVICES=0 %run %t.dir/loop/test
// RUN: llvm-profdata show %t.dir/loop/gfx*.profraw | FileCheck %s --check-prefix=LOOP
// RUN: llvm-profdata show %t.dir/loop/host.profraw | FileCheck %s --check-prefix=NEITHER
// RUN: mkdir %t.dir/both
// RUN: %clang -x hip -O2 -Xarch_device -nogpulib -Xarch_device -nogpuinc \
// RUN:   -fprofile-generate --offload-arch=%amdgpu_arch \
// RUN:   --rocm-path=%hip_lib_path/.. %s -L%hip_lib_path -lamdhip64 \
// RUN:   -Xarch_device -mllvm=-pgo-instrument-entry \
// RUN:   -Xarch_device -mllvm=-pgo-instrument-loop-entries \
// RUN:   -o %t.dir/both/test
// RUN: env LLVM_PROFILE_FILE=%t.dir/both/host.profraw \
// RUN:   LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH \
// RUN:   HIP_VISIBLE_DEVICES=0 %run %t.dir/both/test
// RUN: llvm-profdata show %t.dir/both/gfx*.profraw | FileCheck %s --check-prefix=BOTH
// RUN: llvm-profdata show %t.dir/both/host.profraw | FileCheck %s --check-prefix=NEITHER
// RUN: mkdir %t.dir/neither
// RUN: %clang -x hip -O2 -Xarch_device -nogpulib -Xarch_device -nogpuinc \
// RUN:   -fprofile-generate --offload-arch=%amdgpu_arch \
// RUN:   --rocm-path=%hip_lib_path/.. %s -L%hip_lib_path -lamdhip64 \
// RUN:   -Xarch_host -mllvm=-pgo-instrument-entry \
// RUN:   -Xarch_host -mllvm=-pgo-instrument-loop-entries \
// RUN:   -o %t.dir/neither/test
// RUN: env LLVM_PROFILE_FILE=%t.dir/neither/host.profraw \
// RUN:   LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH \
// RUN:   HIP_VISIBLE_DEVICES=0 %run %t.dir/neither/test
// RUN: llvm-profdata show %t.dir/neither/gfx*.profraw | FileCheck %s --check-prefix=NEITHER
// RUN: llvm-profdata show %t.dir/neither/host.profraw | FileCheck %s --check-prefix=BOTH
// RUN: llvm-profdata merge %t.dir/both/gfx*.profraw -o %t.dir/device.profdata
// RUN: %clang -x hip -O2 -Xarch_device -nogpulib -Xarch_device -nogpuinc \
// RUN:   --offload-arch=%amdgpu_arch --offload-device-only \
// RUN:   --rocm-path=%hip_lib_path/.. -fprofile-use=%t.dir/device.profdata \
// RUN:   -Werror=profile-instr-out-of-date -Werror=profile-instr-unprofiled \
// RUN:   -S -emit-llvm %s -o - | FileCheck %s --check-prefix=USE
//
// ENTRY: Instrumentation level: IR  entry_first = 1  instrument_loop_entries = 0
// LOOP: Instrumentation level: IR  entry_first = 0  instrument_loop_entries = 1
// BOTH: Instrumentation level: IR  entry_first = 1  instrument_loop_entries = 1
// NEITHER: Instrumentation level: IR  entry_first = 0  instrument_loop_entries = 0
// USE-LABEL: define{{.*}} @layout_kernel(
// USE-SAME: !prof ![[ENTRY_COUNT:[0-9]+]]
// USE: ![[ENTRY_COUNT]] = !{!"function_entry_count", i64 64}

#ifdef __HIP_DEVICE_COMPILE__
#  define __global__ __attribute__((global))
#else
#  include <hip/hip_runtime.h>
#endif

extern "C" __global__ void layout_kernel(int *Out, int N) {
  unsigned I = 0;
#ifdef __HIP_DEVICE_COMPILE__
  I = __builtin_amdgcn_workitem_id_x();
#endif
  Out[I] = (I & 1) ? N : 2 * N;
}

#ifndef __HIP_DEVICE_COMPILE__
int main() {
  int *Dev, Host[64];
  if (hipMalloc(&Dev, sizeof(Host)) != hipSuccess)
    return 1;
  layout_kernel<<<1, 64>>>(Dev, 10);
  if (hipDeviceSynchronize() != hipSuccess ||
      hipMemcpy(Host, Dev, sizeof(Host), hipMemcpyDeviceToHost) != hipSuccess)
    return 2;
  for (int I = 0; I < 64; ++I)
    if (Host[I] != ((I & 1) ? 10 : 20))
      return 3;
  return hipFree(Dev) != hipSuccess;
}
#endif
