// RUN: %clang_cc1 %s -verify -cl-std=CL1.1 -triple x86_64-unknown-linux-gnu
// RUN: %clang_cc1 %s -verify -cl-std=CL1.2 -triple x86_64-unknown-linux-gnu
// RUN: %clang_cc1 %s -verify -cl-std=CL2.0 -triple x86_64-unknown-linux-gnu

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef __attribute__((ext_vector_type(4))) float float4;
typedef __attribute__((ext_vector_type(4))) double double4;
typedef __attribute__((ext_vector_type(4))) int int4;
typedef __attribute__((ext_vector_type(4))) long long4;

kernel void float_ops() {
  int flaf = 0.0f && 0.0f;
#if __OPENCL_C_VERSION__ < 120
// expected-error@-2{{invalid operands}}
#endif
  int flof = 0.0f || 0.0f;
#if __OPENCL_C_VERSION__ < 120
// expected-error@-2{{invalid operands}}
#endif
  float fbaf = 0.0f & 0.0f; // expected-error {{invalid operands}}
  float fbof = 0.0f | 0.0f; // expected-error {{invalid operands}}
  float fbxf = 0.0f ^ 0.0f; // expected-error {{invalid operands}}
  int flai = 0.0f && 0;
#if __OPENCL_C_VERSION__ < 120
// expected-error@-2{{invalid operands}}
#endif
  int floi = 0.0f || 0;
#if __OPENCL_C_VERSION__ < 120
// expected-error@-2{{invalid operands}}
#endif
  float ibaf = 0 & 0.0f; // expected-error {{invalid operands to binary expression ('int' and 'float')}}
  float ibof = 0 | 0.0f; // expected-error {{invalid operands}}
  float bnf = ~0.0f;// expected-error {{invalid argument type}}
  float lnf = !0.0f;
#if __OPENCL_C_VERSION__ < 120
// expected-error@-2{{invalid argument type}}
#endif
}

kernel void vec_float_ops() {
  float4 f4 = (float4)(0, 0, 0, 0);
  int4 f4laf = f4 && 0.0f;
#if __OPENCL_C_VERSION__ < 120
// expected-error@-2{{invalid operands}}
#endif
  int4 f4lof = f4 || 0.0f;
#if __OPENCL_C_VERSION__ < 120
// expected-error@-2{{invalid operands}}
#endif
  float4 f4baf = f4 & 0.0f; // expected-error {{invalid operands}}
  float4 f4bof = f4 | 0.0f; // expected-error {{invalid operands}}
  float4 f4bxf = f4 ^ 0.0f; // expected-error {{invalid operands}}
  float bnf4 = ~f4; // expected-error {{invalid argument type}}
  int4 lnf4 = !f4;
#if __OPENCL_C_VERSION__ < 120
// expected-error@-2{{invalid argument type}}
#endif
}

kernel void double_ops() {
  int flaf = 0.0 && 0.0;
#if __OPENCL_C_VERSION__ < 120
// expected-error@-2{{invalid operands}}
#endif
  int flof = 0.0 || 0.0;
#if __OPENCL_C_VERSION__ < 120
// expected-error@-2{{invalid operands}}
#endif
  double fbaf = 0.0 & 0.0; // expected-error {{invalid operands}}
  double fbof = 0.0 | 0.0; // expected-error {{invalid operands}}
  double fbxf = 0.0 ^ 0.0; // expected-error {{invalid operands}}
  int flai = 0.0 && 0;
#if __OPENCL_C_VERSION__ < 120
// expected-error@-2{{invalid operands}}
#endif
  int floi = 0.0 || 0;
#if __OPENCL_C_VERSION__ < 120
// expected-error@-2{{invalid operands}}
#endif
  double ibaf = 0 & 0.0; // expected-error {{invalid operands}}
  double ibof = 0 | 0.0; // expected-error {{invalid operands}}
  double bnf = ~0.0; // expected-error {{invalid argument type}}
  double lnf = !0.0;
#if __OPENCL_C_VERSION__ < 120
// expected-error@-2{{invalid argument type}}
#endif
}

kernel void vec_double_ops() {
  double4 f4 = (double4)(0, 0, 0, 0);
  long4 f4laf = f4 && 0.0;
#if __OPENCL_C_VERSION__ < 120
// expected-error@-2{{invalid operands}}
#endif
  long4 f4lof = f4 || 0.0;
#if __OPENCL_C_VERSION__ < 120
// expected-error@-2{{invalid operands}}
#endif
  double4 f4baf = f4 & 0.0; // expected-error {{invalid operands}}
  double4 f4bof = f4 | 0.0; // expected-error {{invalid operands}}
  double4 f4bxf = f4 ^ 0.0; // expected-error {{invalid operands}}
  double bnf4 = ~f4; // expected-error {{invalid argument type}}
  long4 lnf4 = !f4;
#if __OPENCL_C_VERSION__ < 120
// expected-error@-2{{invalid argument type}}
#endif
}

kernel void pointer_ops(){
  global int* p;
  bool b = !p;
  b = p==0;
  int i;
  b = !&i;
  b = &i==(int *)1;
}
