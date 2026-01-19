// RUN: %clang_cc1 %s -triple spir64 -cl-std=CL2.0 -finclude-default-header -verify -pedantic -fsyntax-only
// RUN: %clang_cc1 %s -triple amdgcn-unknown-unknown -cl-std=CL2.0 -finclude-default-header -verify -pedantic -fsyntax-only

// expected-no-diagnostics

int test_read_1(read_only image2d_t im2d) {
  return get_image_height(im2d);
}

int test_read_2(read_only image3d_t im3d) {
  return get_image_height(im3d);
}

int test_read_3(read_only image2d_array_t im2d_arr) {
  return get_image_height(im2d_arr);
}

int test_read_4(read_only image2d_depth_t im2d_depth) {
  return get_image_height(im2d_depth);
}

int test_read_5(read_only image2d_array_depth_t im2d_arr_depth) {
  return get_image_height(im2d_arr_depth);
}

#ifdef cl_khr_gl_msaa_sharing
int test_read_6(read_only image2d_msaa_t im2d_msaa) {
  return get_image_height(im2d_msaa);
}

int test_read_7(read_only image2d_msaa_depth_t im2d_msaa_depth) {
  return get_image_height(im2d_msaa_depth);
}

int test_read_8(read_only image2d_array_msaa_t im2d_arr_msaa) {
  return get_image_height(im2d_arr_msaa);
}

int test_read_9(read_only image2d_array_msaa_depth_t im2d_arr_msaa_depth) {
  return get_image_height(im2d_arr_msaa_depth);
}
#endif // cl_khr_gl_msaa_sharing

int test_write_1(write_only image2d_t im2d) {
  return get_image_height(im2d);
}

int test_write_2(write_only image3d_t im3d) {
  return get_image_height(im3d);
}

int test_write_3(write_only image2d_array_t im2d_arr) {
  return get_image_height(im2d_arr);
}

int test_write_4(write_only image2d_depth_t im2d_depth) {
  return get_image_height(im2d_depth);
}

int test_write_5(write_only image2d_array_depth_t im2d_arr_depth) {
  return get_image_height(im2d_arr_depth);
}

#ifdef cl_khr_gl_msaa_sharing
int test_write_6(write_only image2d_msaa_t im2d_msaa) {
  return get_image_height(im2d_msaa);
}

int test_write_7(write_only image2d_msaa_depth_t im2d_msaa_depth) {
  return get_image_height(im2d_msaa_depth);
}

int test_write_8(write_only image2d_array_msaa_t im2d_arr_msaa) {
  return get_image_height(im2d_arr_msaa);
}

int test_write_9(write_only image2d_array_msaa_depth_t im2d_arr_msaa_depth) {
  return get_image_height(im2d_arr_msaa_depth);
}
#endif // cl_khr_gl_msaa_sharing

int test_read_write_1(read_write image2d_t im2d) {
  return get_image_height(im2d);
}

int test_read_write_2(read_write image3d_t im3d) {
  return get_image_height(im3d);
}

int test_read_write_3(read_write image2d_array_t im2d_arr) {
  return get_image_height(im2d_arr);
}

int test_read_write_4(read_write image2d_depth_t im2d_depth) {
  return get_image_height(im2d_depth);
}

int test_read_write_5(read_write image2d_array_depth_t im2d_arr_depth) {
  return get_image_height(im2d_arr_depth);
}

#ifdef cl_khr_gl_msaa_sharing
int test_read_write_6(read_write image2d_msaa_t im2d_msaa) {
  return get_image_height(im2d_msaa);
}

int test_read_write_7(read_write image2d_msaa_depth_t im2d_msaa_depth) {
  return get_image_height(im2d_msaa_depth);
}

int test_read_write_8(read_write image2d_array_msaa_t im2d_arr_msaa) {
  return get_image_height(im2d_arr_msaa);
}

int test_read_write_9(read_write image2d_array_msaa_depth_t im2d_arr_msaa_depth) {
  return get_image_height(im2d_arr_msaa_depth);
}
#endif // cl_khr_gl_msaa_sharing
