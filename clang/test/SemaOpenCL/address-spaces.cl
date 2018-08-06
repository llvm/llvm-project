// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only
// RUN: %clang_cc1 %s -cl-std=CL2.0 -verify -pedantic -fsyntax-only
// RUN: %clang_cc1 %s -cl-std=c++ -verify -pedantic -fsyntax-only

__constant int ci = 1;

__kernel void foo(__global int *gip) {
  __local int li;
  __local int lj = 2; // expected-error {{'__local' variable cannot have an initializer}}

  int *ip;
// FIXME: Temporarily disable part of the test that doesn't work for C++ yet.
#if !__OPENCL_CPP_VERSION__
#if __OPENCL_C_VERSION__ < 200
  ip = gip; // expected-error {{assigning '__global int *' to 'int *' changes address space of pointer}}
  ip = &li; // expected-error {{assigning '__local int *' to 'int *' changes address space of pointer}}
  ip = &ci; // expected-error {{assigning '__constant int *' to 'int *' changes address space of pointer}}
#else
  ip = gip;
  ip = &li;
  ip = &ci; // expected-error {{assigning '__constant int *' to '__generic int *' changes address space of pointer}}
#endif
}

void explicit_cast(global int* g, local int* l, constant int* c, private int* p, const constant int *cc)
{
  g = (global int*) l;    // expected-error {{casting '__local int *' to type '__global int *' changes address space of pointer}}
  g = (global int*) c;    // expected-error {{casting '__constant int *' to type '__global int *' changes address space of pointer}}
  g = (global int*) cc;   // expected-error {{casting 'const __constant int *' to type '__global int *' changes address space of pointer}}
  g = (global int*) p;    // expected-error {{casting 'int *' to type '__global int *' changes address space of pointer}}

  l = (local int*) g;     // expected-error {{casting '__global int *' to type '__local int *' changes address space of pointer}}
  l = (local int*) c;     // expected-error {{casting '__constant int *' to type '__local int *' changes address space of pointer}}
  l = (local int*) cc;    // expected-error {{casting 'const __constant int *' to type '__local int *' changes address space of pointer}}
  l = (local int*) p;     // expected-error {{casting 'int *' to type '__local int *' changes address space of pointer}}

  c = (constant int*) g;  // expected-error {{casting '__global int *' to type '__constant int *' changes address space of pointer}}
  c = (constant int*) l;  // expected-error {{casting '__local int *' to type '__constant int *' changes address space of pointer}}
  c = (constant int*) p;  // expected-error {{casting 'int *' to type '__constant int *' changes address space of pointer}}

  p = (private int*) g;   // expected-error {{casting '__global int *' to type 'int *' changes address space of pointer}}
  p = (private int*) l;   // expected-error {{casting '__local int *' to type 'int *' changes address space of pointer}}
  p = (private int*) c;   // expected-error {{casting '__constant int *' to type 'int *' changes address space of pointer}}
  p = (private int*) cc;  // expected-error {{casting 'const __constant int *' to type 'int *' changes address space of pointer}}
}

void ok_explicit_casts(global int *g, global int* g2, local int* l, local int* l2, private int* p, private int* p2)
{
  g = (global int*) g2;
  l = (local int*) l2;
  p = (private int*) p2;
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
  __local __private int var1;   // expected-error {{multiple address spaces specified for type}}
  __local __private int *var2;  // expected-error {{multiple address spaces specified for type}}
  __local private_int_t var3;   // expected-error {{multiple address spaces specified for type}}
  __local private_int_t *var4;  // expected-error {{multiple address spaces specified for type}}
  __private private_int_t var5; // expected-warning {{multiple identical address spaces specified for type}}
  __private private_int_t *var6;// expected-warning {{multiple identical address spaces specified for type}}
#endif // !__OPENCL_CXX_VERSION__
}
