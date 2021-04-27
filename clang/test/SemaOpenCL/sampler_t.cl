// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only
// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -DCHECK_SAMPLER_VALUE -Wspir-compat -triple amdgcn--amdhsa
// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -DCHECK_SAMPLER_VALUE -Wspir-compat -triple spir-unknown-unknown
// RUN: %clang_cc1 %s -cl-std=CL2.0 -verify -pedantic -fsyntax-only
// RUN: %clang_cc1 %s -cl-std=CL2.0 -verify -pedantic -fsyntax-only -DCHECK_SAMPLER_VALUE -Wspir-compat -triple amdgcn--amdhsa
// RUN: %clang_cc1 %s -cl-std=CL2.0 -verify -pedantic -fsyntax-only -DCHECK_SAMPLER_VALUE -Wspir-compat -triple spir-unknown-unknown

#define CLK_ADDRESS_CLAMP_TO_EDGE       2
#define CLK_NORMALIZED_COORDS_TRUE      1
#define CLK_FILTER_NEAREST              0x10
#define CLK_FILTER_LINEAR               0x20

typedef float float4 __attribute__((ext_vector_type(4)));
float4 read_imagef(read_only image1d_t, sampler_t, float);

constant sampler_t glb_smp = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR;
constant sampler_t glb_smp2; // expected-error{{variable in constant address space must be initialized}}
global sampler_t glb_smp3 = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_NEAREST; // expected-error{{sampler type cannot be used with the __local and __global address space qualifiers}} expected-error {{global sampler requires a const or constant address space qualifier}}
const global sampler_t glb_smp3_const = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR;  // expected-error{{sampler type cannot be used with the __local and __global address space qualifiers}}

constant sampler_t glb_smp4 = 0;
#ifdef CHECK_SAMPLER_VALUE
// expected-warning@-2{{sampler initializer has invalid Filter Mode bits}}
#endif

constant sampler_t glb_smp5 = 0x1f;
#ifdef CHECK_SAMPLER_VALUE
// expected-warning@-2{{sampler initializer has invalid Addressing Mode bits}}
#endif

constant sampler_t glb_smp6 = glb_smp; // expected-error{{initializer element is not a compile-time constant}}

int f(void);
constant sampler_t glb_smp7 = f(); // expected-error{{initializer element is not a compile-time constant}}

constant sampler_t glb_smp8 = 1.0f; // expected-error{{initializing '__constant sampler_t' with an expression of incompatible type 'float'}}

constant sampler_t glb_smp9 = 0x100000000LL; // expected-error{{sampler_t initialization requires 32-bit integer, not 'long long'}}

void foo(sampler_t); // expected-note{{passing argument to parameter here}}

void constant_sampler(constant sampler_t s); // expected-error{{parameter may not be qualified with an address space}}

constant struct sampler_s {
  sampler_t smp; // expected-error{{the 'sampler_t' type cannot be used to declare a structure or union field}}
} sampler_str = {0};

sampler_t bad(void); //expected-error{{declaring function return value of type 'sampler_t' is not allowed}}

sampler_t global_nonconst_smp = 0; // expected-error {{global sampler requires a const or constant address space qualifier}}

const sampler_t glb_smp10 = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR;
const constant sampler_t glb_smp11 = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR;

void kernel ker(sampler_t argsmp) {
  local sampler_t smp; // expected-error{{sampler type cannot be used with the __local and __global address space qualifiers}}
  const sampler_t const_smp5 = 1.0f; // expected-error{{initializing 'const sampler_t' with an expression of incompatible type 'float'}}
  const sampler_t const_smp6 = 0x100000000LL; // expected-error{{sampler_t initialization requires 32-bit integer, not 'long long'}}

  foo(5.0f); // expected-error {{passing 'float' to parameter of incompatible type 'sampler_t'}}
  sampler_t sa[] = {argsmp, glb_smp}; // expected-error {{array of 'sampler_t' type is invalid in OpenCL}}
}

#if __OPENCL_C_VERSION__ == 200
void bad(sampler_t *); // expected-error{{pointer to type 'sampler_t' is invalid in OpenCL}}
#else
void bad(sampler_t*); // expected-error{{pointer to type 'sampler_t' is invalid in OpenCL}}
#endif

void bar() {
  sampler_t smp1 = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR;
  sampler_t smp2 = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_NEAREST;
  smp1=smp2; //expected-error{{invalid operands to binary expression ('sampler_t' and 'sampler_t')}}
  smp1+1; //expected-error{{invalid operands to binary expression ('sampler_t' and 'int')}}
  &smp1; //expected-error{{invalid argument type 'sampler_t' to unary expression}}
  *smp2; //expected-error{{invalid argument type 'sampler_t' to unary expression}}
  foo(smp1+1); //expected-error{{invalid operands to binary expression ('sampler_t' and 'int')}}
}

void smp_args(read_only image1d_t image) {
  // Test that parentheses around sampler arguments are ignored.
  float4 res = read_imagef(image, (glb_smp10), 0.0f);
}
