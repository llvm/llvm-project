// NOTE: Do not autogenerate. Check integrated-assembler diagnostics and driver warning controls.
// REQUIRES: amdgpu-registered-target
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -c -o %t.o %s 2>&1 | FileCheck %s --check-prefix=WARN --implicit-check-not=warning:
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -c -o %t.o -Wa,--no-warn %s 2>&1 | count 0
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -c -o %t.o -mllvm -amdgpu-asm-warn-async-barrier-replay=false %s 2>&1 | count 0
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -c -o %t.o -mllvm -amdgpu-asm-warn-async-barrier-replay=true %s 2>&1 | FileCheck %s --check-prefix=WARN --implicit-check-not=warning:
// RUN: not %clang --target=amdgpu12.50-amd-amdhsa -c -o %t.o -Wa,--fatal-warnings %s 2>&1 | FileCheck %s --check-prefix=ERROR --implicit-check-not=warning:
// RUN: echo '.amdgcn_target "amdgpu12.50-amd-amdhsa-unknown-gfx1250:xnack-"' > %t.s
// RUN: cat %s >> %t.s
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -c -o %t.o %t.s 2>&1 | count 0

ds_atomic_async_barrier_arrive_b64 v0
// WARN: :[[@LINE-1]]:1: warning: no recognized replay workaround after 'ds_atomic_async_barrier_arrive_b64' on this GPU
// ERROR: :[[@LINE-2]]:1: error: no recognized replay workaround after 'ds_atomic_async_barrier_arrive_b64' on this GPU
s_endpgm
