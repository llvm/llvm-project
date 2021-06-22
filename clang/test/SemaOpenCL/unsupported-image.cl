// RUN: %clang_cc1 -triple spir-unknown-unknown -verify -cl-std=CL3.0 -cl-ext=-__opencl_c_images %s
// RUN: %clang_cc1 -triple spir-unknown-unknown -verify -cl-std=CL3.0 -cl-ext=+__opencl_c_images %s

#ifdef __opencl_c_images
//expected-no-diagnostics
#endif

void test1(image1d_t i) {}
#if !defined(__opencl_c_images)
// expected-error@-2{{use of type '__read_only image1d_t' requires __opencl_c_images support}}
#endif

void test2(image2d_t i) {}
#if !defined(__opencl_c_images)
// expected-error@-2{{use of type '__read_only image2d_t' requires __opencl_c_images support}}
#endif

void test3(image1d_array_t i) {}
#if !defined(__opencl_c_images)
// expected-error@-2{{use of type '__read_only image1d_array_t' requires __opencl_c_images support}}
#endif

void test4(image2d_array_t i) {}
#if !defined(__opencl_c_images)
// expected-error@-2{{use of type '__read_only image2d_array_t' requires __opencl_c_images support}}
#endif

void test5(image2d_depth_t i) {}
#if !defined(__opencl_c_images)
// expected-error@-2{{use of type '__read_only image2d_depth_t' requires __opencl_c_images support}}
#endif

void test6(image1d_buffer_t i) {}
#if !defined(__opencl_c_images)
// expected-error@-2{{use of type '__read_only image1d_buffer_t' requires __opencl_c_images support}}
#endif

void test7(image2d_msaa_t i) {}
#if !defined(__opencl_c_images)
// expected-error@-2{{use of type '__read_only image2d_msaa_t' requires __opencl_c_images support}}
#endif

void test8(image2d_array_msaa_t i) {}
#if !defined(__opencl_c_images)
// expected-error@-2{{use of type '__read_only image2d_array_msaa_t' requires __opencl_c_images support}}
#endif

void test9(image2d_msaa_depth_t i) {}
#if !defined(__opencl_c_images)
// expected-error@-2{{use of type '__read_only image2d_msaa_depth_t' requires __opencl_c_images support}}
#endif

void test10(image2d_array_msaa_depth_t i) {}
#if !defined(__opencl_c_images)
// expected-error@-2{{use of type '__read_only image2d_array_msaa_depth_t' requires __opencl_c_images support}}
#endif

void test11(sampler_t s) {}
#if !defined(__opencl_c_images)
// expected-error@-2{{use of type 'sampler_t' requires __opencl_c_images support}}
#endif
