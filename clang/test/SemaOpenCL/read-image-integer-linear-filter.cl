// RUN: %clang_cc1 %s -verify -fsyntax-only -cl-std=CL2.0 -fdeclare-opencl-builtins -finclude-default-header
// RUN: %clang_cc1 %s -verify=nowarn -fsyntax-only -cl-std=CL2.0 -fdeclare-opencl-builtins -finclude-default-header -Wno-spir-compat
// nowarn-no-diagnostics

// OpenCL spec: read_imagei and read_imageui support nearest filter only.
// CLK_FILTER_LINEAR in the sampler results in undefined behavior; warn.

// Program scope samplers.
__constant sampler_t glb_linear =
    CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR | CLK_ADDRESS_MIRRORED_REPEAT;
__constant sampler_t glb_nearest =
    CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;

kernel void test_read_imageui_global_sampler(read_only image2d_t img) {
  int2 coord = (int2)(0, 0);
  uint4 u = read_imageui(img, glb_linear, coord); // expected-warning{{'read_imageui' sampler must use CLK_FILTER_NEAREST}}
  u = read_imageui(img, glb_nearest, coord); // no warning
}

kernel void test_read_imagei_global_sampler(read_only image2d_t img) {
  int2 coord = (int2)(0, 0);
  int4 i = read_imagei(img, glb_linear, coord); // expected-warning{{'read_imagei' sampler must use CLK_FILTER_NEAREST}}
}

kernel void test_read_imageui_local_constant(read_only image2d_t img) {
  __constant sampler_t s_linear =
      CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP;
  int2 coord = (int2)(0, 0);
  uint4 u = read_imageui(img, s_linear, coord); // expected-warning{{'read_imageui' sampler must use CLK_FILTER_NEAREST}}
}

kernel void test_read_imageui_nearest_constant(read_only image2d_t img) {
  __constant sampler_t s_nearest =
      CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_NONE;
  int2 coord = (int2)(0, 0);
  uint4 u = read_imageui(img, s_nearest, coord); // no warning
}

kernel void test_read_imageui_local(read_only image2d_t img) {
  sampler_t s_linear =
      CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP;
  int2 coord = (int2)(0, 0);
  uint4 u = read_imageui(img, s_linear, coord); // expected-warning{{'read_imageui' sampler must use CLK_FILTER_NEAREST}}
}

kernel void test_read_imageui_nearest(read_only image2d_t img) {
  sampler_t s_nearest =
      CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_NONE;
  int2 coord = (int2)(0, 0);
  uint4 u = read_imageui(img, s_nearest, coord); // no warning
}

kernel void test_read_imageui_literal(read_only image2d_t img) {
  int2 coord = (int2)(0, 0);
  // CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP = 0x21
  uint4 u = read_imageui(img, CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP, coord); // expected-warning{{'read_imageui' sampler must use CLK_FILTER_NEAREST}}
  // CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_NONE = 0x10
  u = read_imageui(img, CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_NONE, coord); // no warning
}

kernel void test_read_imageui_parameter(read_only image2d_t img, sampler_t smp) {
  int2 coord = (int2)(0, 0);
  uint4 u = read_imageui(img, smp, coord); // no warning
}

kernel void test_read_imagef_linear(read_only image2d_t img) {
  // read_imagef supports linear filtering: no warning.
  float2 coord = (float2)(0.5f, 0.5f);
  float4 f = read_imagef(img, glb_linear, coord); // no warning
}

// Samplerless 1D image reads: integer coordinate must not be mistaken for a
// sampler value even when it looks like CLK_FILTER_LINEAR (e.g. 0x20).
kernel void test_read_imageui_samplerless(read_only image1d_t img) {
  uint4 u = read_imageui(img, 0x10); // no warning
  u = read_imageui(img, 0x20); // no warning
  u = read_imageui(img, 0x30); // no warning
  int4 i = read_imagei(img, 0x10); // no warning
  i = read_imagei(img, 0x20); // no warning
}
