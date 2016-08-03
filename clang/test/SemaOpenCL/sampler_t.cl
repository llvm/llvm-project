// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only

constant sampler_t glb_smp = 5;

void foo(sampler_t);

constant struct sampler_s {
  sampler_t smp; // expected-error{{the 'sampler_t' type cannot be used to declare a structure or union field}}
} sampler_str = {0};

void kernel ker(sampler_t argsmp) {
  local sampler_t smp; // expected-error{{sampler type cannot be used with the __local and __global address space qualifiers}}
  const sampler_t const_smp = 7;
  foo(glb_smp);
  foo(const_smp);
  foo(5); // expected-error{{sampler_t variable required - got 'int'}}
  sampler_t sa[] = {argsmp, const_smp}; // expected-error {{array of 'sampler_t' type is invalid in OpenCL}}
}

void bad(sampler_t*); // expected-error{{pointer to type 'sampler_t' is invalid in OpenCL}}

void bar() {
  sampler_t smp1 = 7;
  sampler_t smp2 = 2;
  smp1=smp2; //expected-error{{invalid operands to binary expression ('sampler_t' and 'sampler_t')}}
  smp1+1; //expected-error{{invalid operands to binary expression ('sampler_t' and 'int')}}
  &smp1; //expected-error{{invalid argument type 'sampler_t' to unary expression}}
  *smp2; //expected-error{{invalid argument type 'sampler_t' to unary expression}}
}

sampler_t bad(); //expected-error{{declaring function return value of type 'sampler_t' is not allowed}}
