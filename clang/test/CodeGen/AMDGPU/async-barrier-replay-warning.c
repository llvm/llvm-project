// NOTE: Do not autogenerate. Check raw inline-assembly warnings and Clang suppression in object emission.
// REQUIRES: amdgpu-registered-target
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nogpulib -c -o %t.o %s 2>&1 | FileCheck %s --check-prefix=WARN --implicit-check-not=warning:
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nogpulib -c -o %t.o -Wa,--no-warn %s 2>&1 | count 0
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -nogpulib -c -o %t.o -mllvm -amdgpu-asm-warn-async-barrier-replay=false %s 2>&1 | count 0
// RUN: %clang --target=amdgpu12.50-amd-amdhsa -mcpu=gfx1250:xnack- -nogpulib -c -o %t.o %s 2>&1 | count 0

void unprotected(void) {
  __asm__ volatile("ds_atomic_async_barrier_arrive_b64 v0" ::: "v0", "memory");
// WARN: warning: no recognized replay workaround after 'ds_atomic_async_barrier_arrive_b64' on this GPU
}

void protected(void) {
  __asm__ volatile("ds_atomic_async_barrier_arrive_b64 v0\n"
                   "s_mov_b32 s7, exec_lo\n"
                   "s_mov_b32 exec_lo, 0\n"
                   "global_load_async_to_lds_b8 v0, v[0:1], off\n"
                   "s_mov_b32 exec_lo, s7"
                   ::: "v0", "v1", "s7", "memory");
}
