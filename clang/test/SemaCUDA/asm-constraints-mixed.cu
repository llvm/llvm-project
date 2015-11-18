// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -triple nvptx-unknown-cuda -fsyntax-only -fcuda-is-device -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify %s

__attribute__((device)) register long global_dev_reg asm("r0");
__attribute__((device)) register long
    global_dev_hreg asm("rsp"); // device-side error

register long global_host_reg asm("rsp");
register long global_host_dreg asm("r0"); // host-side error

__attribute__((device)) void df() {
  register long local_dev_reg asm("r0");
  register long local_host_reg asm("rsp"); // device-side error
  short h;
  // asm with PTX constraints. Some of them are PTX-specific.
  __asm__("dont care" : "=h"(h) : "f"(0.0), "d"(0.0), "h"(0), "r"(0), "l"(0));
}

void hf() {
  register long local_dev_reg asm("r0"); // host-side error
  register long local_host_reg asm("rsp");
  int a;
  // Asm with x86 constraints and registers that are not supported by PTX.
  __asm__("dont care" : "=a"(a) : "a"(0), "b"(0), "c"(0) : "flags");
}

// Check errors in named register variables.
// We should only see errors relevant to current compilation mode.
#if defined(__CUDA_ARCH__)
// Device-side compilation:
// expected-error@8 {{unknown register name 'rsp' in asm}}
// expected-error@15 {{unknown register name 'rsp' in asm}}
#else
// Host-side compilation:
// expected-error@11 {{unknown register name 'r0' in asm}}
// expected-error@22 {{unknown register name 'r0' in asm}}
#endif
