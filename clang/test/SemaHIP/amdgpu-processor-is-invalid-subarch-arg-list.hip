// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu10.10-amd-amdhsa -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple amdgpu9-amd-amdhsa -fsyntax-only -verify %s

__attribute__((device)) void f() {
  // The note lists all processors, not just those covered by the subarch.
  if (__builtin_amdgcn_processor_is("not-a-gpu"))
    // expected-error@-1 {{the argument to __builtin_amdgcn_processor_is must be a valid AMDGCN processor identifier; 'not-a-gpu' is not valid}}
    // expected-note-re@-2 {{valid AMDGCN processor identifiers are: {{.*gfx900.*gfx902.*gfx1010.*gfx1100.*}}}}
    return;
}
