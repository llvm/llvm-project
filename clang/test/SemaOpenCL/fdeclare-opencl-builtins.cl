// RUN: %clang_cc1 %s -triple spir -verify -pedantic -Wconversion -Werror -fsyntax-only -cl-std=CL -fdeclare-opencl-builtins -DNO_HEADER
// RUN: %clang_cc1 %s -triple spir -verify -pedantic -Wconversion -Werror -fsyntax-only -cl-std=CL -fdeclare-opencl-builtins -finclude-default-header
// RUN: %clang_cc1 %s -triple spir -verify -pedantic -Wconversion -Werror -fsyntax-only -cl-std=CL1.2 -fdeclare-opencl-builtins -DNO_HEADER
// RUN: %clang_cc1 %s -triple spir -verify -pedantic -Wconversion -Werror -fsyntax-only -cl-std=CL1.2 -fdeclare-opencl-builtins -finclude-default-header
// RUN: %clang_cc1 %s -triple spir -verify -pedantic -Wconversion -Werror -fsyntax-only -cl-std=CL2.0 -fdeclare-opencl-builtins -DNO_HEADER
// RUN: %clang_cc1 %s -triple spir -verify -pedantic -Wconversion -Werror -fsyntax-only -cl-std=CL2.0 -fdeclare-opencl-builtins -finclude-default-header
// RUN: %clang_cc1 %s -triple spir -verify -pedantic -Wconversion -Werror -fsyntax-only -cl-std=CL3.0 -fdeclare-opencl-builtins -finclude-default-header
// RUN: %clang_cc1 %s -triple spir -verify -pedantic -Wconversion -Werror -fsyntax-only -cl-std=CLC++ -fdeclare-opencl-builtins -DNO_HEADER
// RUN: %clang_cc1 %s -triple spir -verify -pedantic -Wconversion -Werror -fsyntax-only -cl-std=CLC++ -fdeclare-opencl-builtins -finclude-default-header
// RUN: %clang_cc1 %s -triple spir -verify -pedantic -Wconversion -Werror -fsyntax-only -cl-std=CLC++2021 -fdeclare-opencl-builtins -finclude-default-header
// RUN: %clang_cc1 %s -triple spir -verify -pedantic -Wconversion -Werror -fsyntax-only -cl-std=CL2.0 -fdeclare-opencl-builtins -finclude-default-header -cl-ext=-cl_khr_fp64 -DNO_FP64

// Test the -fdeclare-opencl-builtins option.  This is not a completeness
// test, so it should not test for all builtins defined by OpenCL.  Instead
// this test should cover different functional aspects of the TableGen builtin
// function machinery.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#if __OPENCL_C_VERSION__ < CL_VERSION_1_2
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#if __OPENCL_C_VERSION__ <= CL_VERSION_1_2
#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable
#endif

// First, test that Clang gracefully handles missing types.
#ifdef NO_HEADER
void test_without_header() {
  barrier(0);
  // expected-note@-1 0+{{candidate function not viable}}
  // expected-error@-2 0+{{argument type 'void' is incomplete}}
  // expected-error@-3 0+{{no matching function for call to 'barrier'}}
  // expected-error@* {{typedef type cl_mem_fence_flags not found; include the base header with -finclude-default-header}}
}
#endif

// Provide typedefs when invoking clang without -finclude-default-header.
#ifdef NO_HEADER
typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef unsigned short ushort;
typedef __SIZE_TYPE__ size_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;
typedef __INTPTR_TYPE__ intptr_t;
typedef __UINTPTR_TYPE__ uintptr_t;
typedef char char2 __attribute__((ext_vector_type(2)));
typedef char char4 __attribute__((ext_vector_type(4)));
typedef uchar uchar4 __attribute__((ext_vector_type(4)));
typedef uchar uchar16 __attribute__((ext_vector_type(16)));
typedef float float4 __attribute__((ext_vector_type(4)));
typedef float float16 __attribute__((ext_vector_type(16)));
typedef half half4 __attribute__((ext_vector_type(4)));
typedef int int2 __attribute__((ext_vector_type(2)));
typedef int int4 __attribute__((ext_vector_type(4)));
typedef uint uint2 __attribute__((ext_vector_type(2)));
typedef uint uint4 __attribute__((ext_vector_type(4)));
typedef long long2 __attribute__((ext_vector_type(2)));
typedef long long8 __attribute__((ext_vector_type(8)));
typedef ulong ulong4 __attribute__((ext_vector_type(4)));
typedef short short16 __attribute__((ext_vector_type(16)));
typedef ushort ushort3 __attribute__((ext_vector_type(3)));

typedef int clk_profiling_info;
#define CLK_PROFILING_COMMAND_EXEC_TIME 0x1

typedef uint cl_mem_fence_flags;
#define CLK_GLOBAL_MEM_FENCE 0x02

typedef struct {int a;} ndrange_t;

// Enable extensions that are enabled in opencl-c-base.h.
#if (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200)
#define __opencl_c_generic_address_space 1
#define cl_khr_subgroup_extended_types 1
#define cl_khr_subgroup_ballot 1
#define cl_khr_subgroup_non_uniform_arithmetic 1
#define cl_khr_subgroup_clustered_reduce 1
#define __opencl_c_read_write_images 1
#endif

#define __opencl_c_named_address_space_builtins 1
#endif

kernel void test_pointers(volatile global void *global_p, global const int4 *a) {
  int i;
  unsigned int ui;

  prefetch(a, 2);

  atom_add((volatile __global int *)global_p, i);
  atom_cmpxchg((volatile __global unsigned int *)global_p, ui, ui);
}

// Only test enum arguments when the base header is included, because we need
// the enum declarations.
#if !defined(NO_HEADER) && (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200)
kernel void test_enum_args(volatile global atomic_int *global_p, global int *expected) {
  int desired;
  atomic_compare_exchange_strong_explicit(global_p, expected, desired,
                                          memory_order_acq_rel,
                                          memory_order_relaxed,
                                          memory_scope_work_group);
}
#endif

#if defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200
void test_typedef_args(clk_event_t evt, volatile atomic_flag *flg, global unsigned long long *values) {
  capture_event_profiling_info(evt, CLK_PROFILING_COMMAND_EXEC_TIME, values);

  atomic_flag_clear(flg);
  bool result = atomic_flag_test_and_set(flg);

  size_t ws[2] = {2, 8};
  ndrange_t r = ndrange_2D(ws);
}

// Check that atomic_fetch_ functions can be called with (u)intptr_t arguments,
// despite OpenCLBuiltins.td not providing explicit overloads for those types.
void test_atomic_fetch(volatile __generic atomic_int *a_int,
                       volatile __generic atomic_intptr_t *a_intptr,
                       volatile __generic atomic_uintptr_t *a_uintptr) {
  int i;
  intptr_t ip;
  uintptr_t uip;
  ptrdiff_t ptrdiff;

  i = atomic_fetch_add(a_int, i);
  ip = atomic_fetch_add(a_intptr, ptrdiff);
  uip = atomic_fetch_add(a_uintptr, ptrdiff);

  ip = atomic_fetch_or(a_intptr, ip);
  uip = atomic_fetch_or(a_uintptr, uip);
}
#endif

#if !defined(NO_HEADER) && !defined(NO_FP64) && __OPENCL_C_VERSION__ >= 200
// Check added atomic_fetch_ functions by cl_ext_float_atomics
// extension can be called
void test_atomic_fetch_with_address_space(volatile __generic atomic_float *a_float,
                                          volatile __generic atomic_double *a_double,
                                          volatile __local atomic_float *a_float_local,
                                          volatile __local atomic_double *a_double_local,
                                          volatile __global atomic_float *a_float_global,
                                          volatile __global atomic_double *a_double_global) {
  float f1, resf1;
  double d1, resd1;
  resf1 = atomic_fetch_min(a_float, f1);
  resf1 = atomic_fetch_max_explicit(a_float_local, f1, memory_order_seq_cst);
  resf1 = atomic_fetch_add_explicit(a_float_global, f1, memory_order_seq_cst, memory_scope_work_group);

  resd1 = atomic_fetch_min(a_double, d1);
  resd1 = atomic_fetch_max_explicit(a_double_local, d1, memory_order_seq_cst);
  resd1 = atomic_fetch_add_explicit(a_double_global, d1, memory_order_seq_cst, memory_scope_work_group);
}
#endif // !defined(NO_HEADER) && __OPENCL_C_VERSION__ >= 200

// Test old atomic overloaded with generic address space in C++ for OpenCL.
#if __OPENCL_C_VERSION__ >= 200
void test_legacy_atomics_cpp(__generic volatile unsigned int *a) {
  atomic_add(a, 1);
#if !defined(__cplusplus)
  // expected-error@-2{{no matching function for call to 'atomic_add'}}
  // expected-note@-3 4 {{candidate function not viable}}
#endif
}
#endif

kernel void basic_conversion() {
  float f;
  char2 c2;
  long2 l2;
  float4 f4;
  int4 i4;

#ifdef NO_FP64
  (void)convert_double_rtp(f);
  // expected-error@-1{{implicit declaration of function 'convert_double_rtp' is invalid in OpenCL}}
#else
  double d;
  f = convert_float(d);
#endif
  l2 = convert_long2_rtz(c2);
  i4 = convert_int4_sat(f4);
}

kernel void basic_conversion_neg() {
  int i;
  float f;

  f = convert_float_sat(i);
#if !defined(__OPENCL_CPP_VERSION__)
  // expected-error@-2{{implicit declaration of function 'convert_float_sat' is invalid in OpenCL}}
  // expected-error@-3{{implicit conversion from 'int' to 'float' may lose precision}}
#else
  // expected-error@-5{{use of undeclared identifier 'convert_float_sat'; did you mean 'convert_float'?}}
  // expected-note@-6{{'convert_float' declared here}}
#endif
}

char4 test_int(char c, char4 c4) {
  char m = max(c, c);
  char4 m4 = max(c4, c4);
  uchar4 abs1 = abs(c4);
  uchar4 abs2 = abs(abs1);
  return max(c4, c);
}

kernel void basic_vector_misc(float4 a) {
  float4 res;
  uint4 mask = (uint4)(1, 2, 3, 4);

  res = shuffle(a, mask);
}

kernel void basic_image_readonly(read_only image2d_t image_read_only_image2d) {
  int2 i2;
  sampler_t sampler;
  half4 res;
  float4 resf;

  resf = read_imagef(image_read_only_image2d, i2);
  res = read_imageh(image_read_only_image2d, i2);
#if __OPENCL_C_VERSION__ < CL_VERSION_1_2 && !defined(__OPENCL_CPP_VERSION__)
  // expected-error@-3{{no matching function for call to 'read_imagef'}}
  // expected-note@-4 + {{candidate function not viable}}
  // expected-error@-4{{no matching function for call to 'read_imageh'}}
  // expected-note@-5 + {{candidate function not viable}}
#endif
  res = read_imageh(image_read_only_image2d, sampler, i2);

  int imgWidth = get_image_width(image_read_only_image2d);
}

#if __OPENCL_C_VERSION__ >= CL_VERSION_2_0
kernel void basic_image_readwrite(read_write image3d_t image_read_write_image3d) {
  half4 h4;
  int4 i4;

  write_imageh(image_read_write_image3d, i4, h4);

  int imgDepth = get_image_depth(image_read_write_image3d);
}
#endif // __OPENCL_C_VERSION__ >= CL_VERSION_2_0

kernel void basic_image_writeonly(write_only image1d_buffer_t image_write_only_image1d_buffer, write_only image3d_t image3dwo) {
  half4 h4;
  float4 f4;
  int i;

  write_imagef(image_write_only_image1d_buffer, i, f4);
  write_imageh(image_write_only_image1d_buffer, i, h4);

  int4 i4;
  write_imagef(image3dwo, i4, i, f4);
#if __OPENCL_C_VERSION__ <= CL_VERSION_1_2 && !defined(__OPENCL_CPP_VERSION__)
  // expected-error@-2{{no matching function for call to 'write_imagef'}}
  // expected-note@-3 + {{candidate function not viable}}
#endif
}

kernel void basic_subgroup(global uint *out) {
  out[0] = get_sub_group_size();
#if __OPENCL_C_VERSION__ <= CL_VERSION_1_2 && !defined(__OPENCL_CPP_VERSION__)
  // expected-error@-2{{implicit declaration of function 'get_sub_group_size' is invalid in OpenCL}}
  // expected-error@-3{{implicit conversion changes signedness}}
#endif

// Only test when the base header is included, because we need the enum declarations.
#if !defined(NO_HEADER) && (defined(__OPENCL_CPP_VERSION__) || __OPENCL_C_VERSION__ >= 200)
  sub_group_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);
#endif
}

kernel void extended_subgroup(global uint4 *out, global int *scalar, global char2 *c2) {
  out[0] = get_sub_group_eq_mask();
  scalar[0] = sub_group_non_uniform_scan_inclusive_or(3);
  scalar[1] = sub_group_clustered_reduce_logical_xor(2, 4);
  *c2 = sub_group_broadcast(*c2, 2);
#if __OPENCL_C_VERSION__ < CL_VERSION_2_0 && !defined(__OPENCL_CPP_VERSION__)
  // expected-error@-5{{implicit declaration of function 'get_sub_group_eq_mask' is invalid in OpenCL}}
  // expected-error@-6{{implicit conversion changes signedness}}
  // expected-error@-6{{implicit declaration of function 'sub_group_non_uniform_scan_inclusive_or' is invalid in OpenCL}}
  // expected-error@-6{{implicit declaration of function 'sub_group_clustered_reduce_logical_xor' is invalid in OpenCL}}
  // expected-error@-6{{implicit declaration of function 'sub_group_broadcast' is invalid in OpenCL}}
  // expected-error@-7{{implicit conversion loses integer precision}}
#endif
}

kernel void basic_vector_data() {
#if __OPENCL_C_VERSION__ >= CL_VERSION_2_0
  generic void *generic_p;
#endif
  constant void *constant_p;
  local void *local_p;
  global void *global_p;
  private void *private_p;
  size_t s;
  ulong4 ul4;
  short16 s16;
#if __OPENCL_C_VERSION__ >= CL_VERSION_2_0
  ushort3 us3;
  uchar16 uc16;
#endif
  long8 l8;
  uint2 ui2;
  float16 f16;

  ul4 = vload4(s, (const __constant ulong *) constant_p);
  s16 = vload16(s, (const __constant short *) constant_p);

#if __OPENCL_C_VERSION__ >= CL_VERSION_2_0
  us3 = vload3(s, (const __generic ushort *) generic_p);
  uc16 = vload16(s, (const __generic uchar *) generic_p);
#endif

  l8 = vload8(s, (const __global long *) global_p);
  ui2 = vload2(s, (const __local uint *) local_p);
  f16 = vload16(s, (const __private float *) private_p);
}

kernel void basic_work_item() {
  uint ui;

  barrier(CLK_GLOBAL_MEM_FENCE);

  get_enqueued_local_size(ui);
#if !defined(__OPENCL_CPP_VERSION__) && __OPENCL_C_VERSION__ < CL_VERSION_2_0
// expected-error@-2{{implicit declaration of function 'get_enqueued_local_size' is invalid in OpenCL}}
#endif
}

#ifdef NO_FP64
void test_extension_types(char2 c2) {
  // We should see 6 candidates for float and half types, and none for double types.
  int i = isnan(c2);
  // expected-error@-1{{no matching function for call to 'isnan'}}
  // expected-note@-2 6 {{candidate function not viable: no known conversion from '__private char2' (vector of 2 'char' values) to 'float}}
  // expected-note@-3 6 {{candidate function not viable: no known conversion from '__private char2' (vector of 2 'char' values) to 'half}}
}
#endif
