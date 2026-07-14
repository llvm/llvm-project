// Test that 64-bit integer types require cles_khr_int64 (OpenCL C 1.x,
// embedded profile) or the __opencl_c_int64 feature (OpenCL C 3.0).

// 64-bit integers are available by default: the full profile is assumed.
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -verify=good -fsyntax-only -cl-std=CL1.2
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -verify=good -fsyntax-only -cl-std=CL3.0

// Either option restores 64-bit integers after -cl-ext=-all.
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -verify=good -fsyntax-only -cl-std=CL1.2 -cl-ext=-all,+cles_khr_int64
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -verify=good -fsyntax-only -cl-std=CL3.0 -cl-ext=-all,+__opencl_c_int64

// With both options disabled, 64-bit integer types and literals are rejected.
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -verify=err12 -fsyntax-only -cl-std=CL1.2 -cl-ext=-cles_khr_int64,-__opencl_c_int64
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -verify=err30 -fsyntax-only -cl-std=CL3.0 -cl-ext=-all

// The -fdeclare-opencl-builtins overloads follow the same two options.
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -verify=good -fsyntax-only -cl-std=CL3.0 -fdeclare-opencl-builtins -DBUILTINS
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -verify=good -fsyntax-only -cl-std=CL1.2 -cl-ext=-all,+cles_khr_int64 -fdeclare-opencl-builtins -DBUILTINS
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -verify=err30,errb -fsyntax-only -cl-std=CL3.0 -cl-ext=-all -fdeclare-opencl-builtins -DBUILTINS

// good-no-diagnostics

typedef long long_vec2 __attribute__((ext_vector_type(2)));
// err12-error@-1{{use of type 'long' requires cles_khr_int64 support}}
// err30-error@-2{{use of type 'long' requires __opencl_c_int64 support}}

kernel void test_int64(void) {
  long l;
  // err12-error@-1{{use of type 'long' requires cles_khr_int64 support}}
  // err30-error@-2{{use of type 'long' requires __opencl_c_int64 support}}

  unsigned long ul;
  // err12-error@-1{{use of type 'unsigned long' requires cles_khr_int64 support}}
  // err30-error@-2{{use of type 'unsigned long' requires __opencl_c_int64 support}}

  // Pointers to 64-bit integers are rejected as well: without the extension
  // the types do not exist at all (unlike half with cl_khr_fp16).
  private long *p;
  // err12-error@-1{{use of type 'long' requires cles_khr_int64 support}}
  // err30-error@-2{{use of type 'long' requires __opencl_c_int64 support}}

  // A literal that does not fit in 32 bits gets a 64-bit type without any
  // 'long' declaration, so it is diagnosed too.
  (void)5000000000;
  // err12-error@-1{{use of type 'long' requires cles_khr_int64 support}}
  // err30-error@-2{{use of type 'long' requires __opencl_c_int64 support}}

  // 32-bit literals are fine.
  (void)2147483647;
}

#ifdef BUILTINS
kernel void test_int64_builtins(int i) {
  long l = convert_long(i);
  // err30-error@-1{{use of type 'long' requires __opencl_c_int64 support}}
  // errb-error@-2{{use of undeclared identifier 'convert_long'}}
  (void)l;
}
#endif
