// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only
// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -DCHECK_SAMPLER_VALUE -Wspir-compat -triple amdgcn--amdhsa
// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -DCHECK_SAMPLER_VALUE -triple spir-unknown-unknown

#define CLK_ADDRESS_CLAMP_TO_EDGE       2
#define CLK_NORMALIZED_COORDS_TRUE      1
#define CLK_FILTER_NEAREST              0x10
#define CLK_FILTER_LINEAR               0x20

constant sampler_t glb_smp = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR;
constant sampler_t glb_smp2; // expected-error{{variable in constant address space must be initialized}}
global sampler_t glb_smp3 = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_NEAREST; // expected-error{{sampler type cannot be used with the __local and __global address space qualifiers}}

constant sampler_t glb_smp4 = 0;
#ifdef CHECK_SAMPLER_VALUE
// expected-warning@-2{{sampler initializer has invalid Filter Mode bits}}
#endif

constant sampler_t glb_smp5 = 0x1f;
#ifdef CHECK_SAMPLER_VALUE
// expected-warning@-2{{sampler initializer has invalid Addressing Mode bits}}
#endif

void foo(sampler_t);

constant struct sampler_s {
  sampler_t smp; // expected-error{{the 'sampler_t' type cannot be used to declare a structure or union field}}
} sampler_str = {0};

void kernel ker(sampler_t argsmp) {
  local sampler_t smp; // expected-error{{sampler type cannot be used with the __local and __global address space qualifiers}}
  const sampler_t const_smp = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR;
  const sampler_t const_smp2;
  foo(glb_smp);
  foo(glb_smp2);
  foo(glb_smp3);
  foo(const_smp);
  foo(const_smp2);
  foo(argsmp);
  foo(5); // expected-error{{sampler_t variable required - got 'int'}}
  sampler_t sa[] = {argsmp, const_smp}; // expected-error {{array of 'sampler_t' type is invalid in OpenCL}}
}

void bad(sampler_t*); // expected-error{{pointer to type 'sampler_t' is invalid in OpenCL}}

void bar() {
  sampler_t smp1 = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR;
  sampler_t smp2 = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_NEAREST;
  smp1=smp2; //expected-error{{invalid operands to binary expression ('sampler_t' and 'sampler_t')}}
  smp1+1; //expected-error{{invalid operands to binary expression ('sampler_t' and 'int')}}
  &smp1; //expected-error{{invalid argument type 'sampler_t' to unary expression}}
  *smp2; //expected-error{{invalid argument type 'sampler_t' to unary expression}}
}

sampler_t bad(void); //expected-error{{declaring function return value of type 'sampler_t' is not allowed}}
