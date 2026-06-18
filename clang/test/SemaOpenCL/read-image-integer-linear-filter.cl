// RUN: %clang_cc1 %s -verify -fsyntax-only -cl-std=CL2.0 -fdeclare-opencl-builtins -finclude-default-header

// OpenCL spec: read_imagei and read_imageui support nearest filter only.
// CLK_FILTER_LINEAR in the sampler results in undefined behavior; warn.

// Program scope samplers.
__constant sampler_t glb_linear =
    CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR | CLK_ADDRESS_MIRRORED_REPEAT;
__constant sampler_t glb_nearest =
    CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;

kernel void test_read_imageui_global_sampler(read_only image2d_t img, global uint *out) {
  int2 coord = (int2)(0, 0);
  *out = read_imageui(img, glb_linear, coord).s0; // expected-warning{{'read_imageui' sampler must use CLK_FILTER_NEAREST}}
  *out = read_imageui(img, glb_nearest, coord).s0; // no warning
}

kernel void test_read_imagei_global_sampler(read_only image2d_t img, global int *out) {
  int2 coord = (int2)(0, 0);
  *out = read_imagei(img, glb_linear, coord).s0; // expected-warning{{'read_imagei' sampler must use CLK_FILTER_NEAREST}}
}

kernel void test_read_imageui_local_constant(read_only image2d_t img, global uint *out) {
  __constant sampler_t s_linear =
      CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP;
  int2 coord = (int2)(0, 0);
  *out = read_imageui(img, s_linear, coord).s0; // expected-warning{{'read_imageui' sampler must use CLK_FILTER_NEAREST}}
}

kernel void test_read_imageui_nearest_constant(read_only image2d_t img, global uint *out) {
  __constant sampler_t s_nearest =
      CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_NONE;
  int2 coord = (int2)(0, 0);
  *out = read_imageui(img, s_nearest, coord).s0; // no warning
}

kernel void test_read_imageui_local(read_only image2d_t img, global uint *out) {
  sampler_t s_linear =
      CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP;
  int2 coord = (int2)(0, 0);
  *out = read_imageui(img, s_linear, coord).s0; // expected-warning{{'read_imageui' sampler must use CLK_FILTER_NEAREST}}
}

kernel void test_read_imageui_nearest(read_only image2d_t img, global uint *out) {
  sampler_t s_nearest =
      CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_NONE;
  int2 coord = (int2)(0, 0);
  *out = read_imageui(img, s_nearest, coord).s0; // no warning
}

kernel void test_read_imageui_literal(read_only image2d_t img, global uint *out) {
  int2 coord = (int2)(0, 0);
  // CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP = 0x21
  *out = read_imageui(img, CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP, coord).s0; // expected-warning{{'read_imageui' sampler must use CLK_FILTER_NEAREST}}
  // CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_NONE = 0x10
  *out = read_imageui(img, CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_NONE, coord).s0; // no warning
}

kernel void test_read_imageui_parameter(read_only image2d_t img, global uint *out, sampler_t smp) {
  int2 coord = (int2)(0, 0);
  *out = read_imageui(img, smp, coord).s0; // no warning
}

kernel void test_read_imagef_linear(read_only image2d_t img, global float *out) {
  // read_imagef supports linear filtering: no warning.
  float2 coord = (float2)(0.5f, 0.5f);
  *out = read_imagef(img, glb_linear, coord).s0; // no warning
}

// Samplerless 1D image reads: integer coordinate must not be mistaken for a
// sampler value even when it looks like CLK_FILTER_LINEAR (e.g. 0x20).
kernel void test_read_imageui_samplerless(read_only image1d_t img, global uint *out) {
  *out = read_imageui(img, 0x10).s0; // no warning
  *out = read_imageui(img, 0x20).s0; // no warning
  *out = read_imageui(img, 0x30).s0; // no warning
  *out = read_imagei(img, 0x10).s0;  // no warning
  *out = read_imagei(img, 0x20).s0;  // no warning
}
