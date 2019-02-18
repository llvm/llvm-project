// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only
// RUN: %clang_cc1 %s -cl-std=CL2.0 -verify -pedantic -fsyntax-only
// RUN: %clang_cc1 %s -cl-std=c++ -verify -pedantic -fsyntax-only

__constant int ci = 1;

__kernel void foo(__global int *gip) {
  __local int li;
  __local int lj = 2; // expected-error {{'__local' variable cannot have an initializer}}

  int *ip;
#if ((!__OPENCL_CPP_VERSION__) && (__OPENCL_C_VERSION__ < 200))
  ip = gip; // expected-error {{assigning '__global int *' to 'int *' changes address space of pointer}}
  ip = &li; // expected-error {{assigning '__local int *' to 'int *' changes address space of pointer}}
  ip = &ci; // expected-error {{assigning '__constant int *' to 'int *' changes address space of pointer}}
#else
  ip = gip;
  ip = &li;
  ip = &ci;
#if !__OPENCL_CPP_VERSION__
// expected-error@-2 {{assigning '__constant int *' to '__generic int *' changes address space of pointer}}
#else
// expected-error@-4 {{assigning to '__generic int *' from incompatible type '__constant int *'}}
#endif
#endif
}

void explicit_cast(__global int *g, __local int *l, __constant int *c, __private int *p, const __constant int *cc) {
  g = (__global int *)l;  // expected-error {{casting '__local int *' to type '__global int *' changes address space of pointer}}
  g = (__global int *)c;  // expected-error {{casting '__constant int *' to type '__global int *' changes address space of pointer}}
  g = (__global int *)cc; // expected-error {{casting 'const __constant int *' to type '__global int *' changes address space of pointer}}
  g = (__global int *)p;  // expected-error {{casting 'int *' to type '__global int *' changes address space of pointer}}

  l = (__local int *)g;  // expected-error {{casting '__global int *' to type '__local int *' changes address space of pointer}}
  l = (__local int *)c;  // expected-error {{casting '__constant int *' to type '__local int *' changes address space of pointer}}
  l = (__local int *)cc; // expected-error {{casting 'const __constant int *' to type '__local int *' changes address space of pointer}}
  l = (__local int *)p;  // expected-error {{casting 'int *' to type '__local int *' changes address space of pointer}}

  c = (__constant int *)g; // expected-error {{casting '__global int *' to type '__constant int *' changes address space of pointer}}
  c = (__constant int *)l; // expected-error {{casting '__local int *' to type '__constant int *' changes address space of pointer}}
  c = (__constant int *)p; // expected-error {{casting 'int *' to type '__constant int *' changes address space of pointer}}

  p = (__private int *)g;  // expected-error {{casting '__global int *' to type 'int *' changes address space of pointer}}
  p = (__private int *)l;  // expected-error {{casting '__local int *' to type 'int *' changes address space of pointer}}
  p = (__private int *)c;  // expected-error {{casting '__constant int *' to type 'int *' changes address space of pointer}}
  p = (__private int *)cc; // expected-error {{casting 'const __constant int *' to type 'int *' changes address space of pointer}}
}

void ok_explicit_casts(__global int *g, __global int *g2, __local int *l, __local int *l2, __private int *p, __private int *p2) {
  g = (__global int *)g2;
  l = (__local int *)l2;
  p = (__private int *)p2;
}

__private int func_return_priv(void);       //expected-error {{return value cannot be qualified with address space}}
__global int func_return_global(void);      //expected-error {{return value cannot be qualified with address space}}
__local int func_return_local(void);        //expected-error {{return value cannot be qualified with address space}}
__constant int func_return_constant(void);  //expected-error {{return value cannot be qualified with address space}}
#if __OPENCL_C_VERSION__ >= 200
__generic int func_return_generic(void);    //expected-error {{return value cannot be qualified with address space}}
#endif

void func_multiple_addr(void) {
  typedef __private int private_int_t;
  __private __local int var1;   // expected-error {{multiple address spaces specified for type}}
  __private __local int *var2;  // expected-error {{multiple address spaces specified for type}}
  __local private_int_t var3;   // expected-error {{multiple address spaces specified for type}}
  __local private_int_t *var4;  // expected-error {{multiple address spaces specified for type}}
  __private private_int_t var5; // expected-warning {{multiple identical address spaces specified for type}}
  __private private_int_t *var6;// expected-warning {{multiple identical address spaces specified for type}}
}
