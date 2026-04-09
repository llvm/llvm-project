// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -x c++ -fsycl-is-host -verify=sycl %s
// RUN: %clang_cc1 -fsyntax-only -x c++ -fsycl-is-device -verify=sycl %s
// RUN: %clang_cc1 -fsyntax-only -x hlsl -triple dxil-pc-shadermodel6.3-library -verify=hlsl %s

// hlsl-error@#constant{{'opencl_constant' attribute is not supported in HLSL}}
// hlsl-error@#generic{{'opencl_generic' attribute is not supported in HLSL}}
// hlsl-error@#global{{'opencl_global' attribute is not supported in HLSL}}
// hlsl-error@#global_host{{'opencl_global_host' attribute is not supported in HLSL}}
// hlsl-error@#global_device{{'opencl_global_device' attribute is not supported in HLSL}}
// hlsl-error@#local{{'opencl_local' attribute is not supported in HLSL}}
// hlsl-error@#private{{'opencl_private' attribute is not supported in HLSL}}

// sycl-error@#constant{{'opencl_constant' attribute is not supported in SYCL}}
// sycl-error@#generic{{'opencl_generic' attribute is not supported in SYCL}}

int __attribute__((opencl_constant)) glob = 1; // #constant
int __attribute__((opencl_generic)) gen; // #generic
int __attribute__((opencl_global)) c; // #global
int __attribute__((opencl_global_host)) h; // #global_host
int __attribute__((opencl_global_device)) d; // #global_device
int __attribute__((opencl_local)) l; // #local
int __attribute__((opencl_private)) p; // #private

// expected-no-diagnostics
