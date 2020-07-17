// Tests that host and target builtins can be used in the same TU,
// have appropriate host/device attributes and that CUDA call
// restrictions are enforced. Also verify that non-target builtins can
// be used from both host and device functions.
//
// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN:     -aux-triple nvptx64-unknown-cuda \
// RUN:     -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple nvptx64-unknown-cuda -fcuda-is-device \
// RUN:     -aux-triple x86_64-unknown-unknown \
// RUN:     -fsyntax-only -verify %s

#if !(defined(__amd64__) && defined(__PTX__))
#error "Expected to see preprocessor macros from both sides of compilation."
#endif

void hf() {
  int x = __builtin_ia32_rdtsc();
  int y = __nvvm_read_ptx_sreg_tid_x(); // expected-note  {{'__nvvm_read_ptx_sreg_tid_x' declared here}}
  // expected-error@-1 {{reference to __device__ function '__nvvm_read_ptx_sreg_tid_x' in __host__ function}}
  x = __builtin_abs(1);
}

__attribute__((device)) void df() {
  int x = __nvvm_read_ptx_sreg_tid_x();
  int y = __builtin_ia32_rdtsc(); // expected-error {{reference to __host__ function '__builtin_ia32_rdtsc' in __device__ function}}
                                  // expected-note@20 {{'__builtin_ia32_rdtsc' declared here}}
  x = __builtin_abs(1);
}
