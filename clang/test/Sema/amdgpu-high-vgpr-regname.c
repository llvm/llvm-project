// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1250 -fsyntax-only -verify=gfx1250 %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx942 -fsyntax-only -verify=gfx942 %s

// VGPRs v256-v1023 are addressable only on gfx1250+ (1024 addressable VGPRs);
// naming them in inline asm must be rejected on targets with only 256 VGPRs.

// gfx1250-no-diagnostics

void low_vgpr(void) {
  __asm__ volatile("" ::: "v100");
}

void high_vgpr(void) {
  __asm__ volatile("" ::: "v300"); // gfx942-error {{unknown register name 'v300' in asm}}
}
