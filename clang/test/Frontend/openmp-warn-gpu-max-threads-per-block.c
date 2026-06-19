// REQUIRES: amdgpu-registered-target

// Test that --gpu-max-threads-per-block warns when used outside of HIP.
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa --gpu-max-threads-per-block=1024 \
// RUN:     -fsyntax-only -verify=warn %s

// Test that no warning is emitted for HIP.
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device \
// RUN:     --gpu-max-threads-per-block=1024 -fsyntax-only -verify=hip %s

// warn-warning@*{{'--gpu-max-threads-per-block=1024' is ignored since it is only supported for HIP}}
// hip-no-diagnostics

void f(void) {}
