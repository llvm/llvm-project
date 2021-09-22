// RUN: %clang_cc1 -triple spir-unknown-unknown -verify -cl-std=CL3.0 -cl-ext=-__opencl_c_images,-__opencl_c_read_write_images,-cl_khr_3d_image_writes,-__opencl_c_3d_image_writes %s
// RUN: %clang_cc1 -triple spir-unknown-unknown -verify -cl-std=CL3.0 -cl-ext=+__opencl_c_images,+__opencl_c_read_write_images,+cl_khr_3d_image_writes,+__opencl_c_3d_image_writes %s
// RUN: %clang_cc1 -triple spir-unknown-unknown -verify -cl-std=CL3.0 -cl-ext=+__opencl_c_images,+__opencl_c_read_write_images,-cl_khr_3d_image_writes,-__opencl_c_3d_image_writes %s
// RUN: %clang_cc1 -triple spir-unknown-unknown -verify -cl-std=clc++2021 -cl-ext=-__opencl_c_images,-__opencl_c_read_write_images,-cl_khr_3d_image_writes,-__opencl_c_3d_image_writes %s
// RUN: %clang_cc1 -triple spir-unknown-unknown -verify -cl-std=clc++2021 -cl-ext=+__opencl_c_images,+__opencl_c_read_write_images,+cl_khr_3d_image_writes,+__opencl_c_3d_image_writes %s
// RUN: %clang_cc1 -triple spir-unknown-unknown -verify -cl-std=clc++2021 -cl-ext=+__opencl_c_images,+__opencl_c_read_write_images,-cl_khr_3d_image_writes,-__opencl_c_3d_image_writes %s

#if defined(__opencl_c_images) && defined(__opencl_c_3d_image_writes)
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

void test12(write_only image3d_t i) {}
#if !defined(__opencl_c_images)
// expected-error@-2{{use of type '__write_only image3d_t' requires __opencl_c_images support}}
#elif !defined(__opencl_c_3d_image_writes)
// expected-error@-4{{use of type '__write_only image3d_t' requires cl_khr_3d_image_writes and __opencl_c_3d_image_writes support}}
#endif
