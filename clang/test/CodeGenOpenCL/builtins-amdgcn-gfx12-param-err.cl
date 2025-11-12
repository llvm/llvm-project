// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1200 -verify -emit-llvm -o - %s

kernel void builtins_amdgcn_s_barrier_signal_err(global int* in, global int* out, int barrier) {

  __builtin_amdgcn_s_barrier_signal(barrier); // expected-error {{'__builtin_amdgcn_s_barrier_signal' must be a constant integer}}
  __builtin_amdgcn_s_barrier_wait(-1);
  *out = *in;
}

kernel void builtins_amdgcn_s_barrier_wait_err(global int* in, global int* out, int barrier) {

  __builtin_amdgcn_s_barrier_signal(-1);
  __builtin_amdgcn_s_barrier_wait(barrier); // expected-error {{'__builtin_amdgcn_s_barrier_wait' must be a constant integer}}
  *out = *in;
}

kernel void builtins_amdgcn_s_barrier_signal_isfirst_err(global int* in, global int* out, int barrier) {

  __builtin_amdgcn_s_barrier_signal_isfirst(barrier); // expected-error {{'__builtin_amdgcn_s_barrier_signal_isfirst' must be a constant integer}}
  __builtin_amdgcn_s_barrier_wait(-1);
  *out = *in;
}

kernel void builtins_amdgcn_s_barrier_leave_err(global int* in, global int* out, int barrier) {

  __builtin_amdgcn_s_barrier_signal(-1);
  __builtin_amdgcn_s_barrier_leave(barrier); // expected-error {{'__builtin_amdgcn_s_barrier_leave' must be a constant integer}}
  *out = *in;
}

void test_s_buffer_prefetch_data(__amdgpu_buffer_rsrc_t rsrc, unsigned int off)
{
  __builtin_amdgcn_s_buffer_prefetch_data(rsrc, off, 31); // expected-error {{'__builtin_amdgcn_s_buffer_prefetch_data' must be a constant integer}}
}
