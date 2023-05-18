// RUN: %clang_cc1 %s -verify=expected -pedantic -fsyntax-only
// RUN: %clang_cc1 %s -cl-std=CL2.0 -verify=expected -pedantic -fsyntax-only
// RUN: %clang_cc1 %s -cl-std=CL3.0 -cl-ext=+__opencl_c_generic_address_space -verify=expected -pedantic -fsyntax-only
// RUN: %clang_cc1 %s -cl-std=clc++1.0 -verify -pedantic -fsyntax-only
// RUN: %clang_cc1 %s -cl-std=clc++2021 -cl-ext=+__opencl_c_generic_address_space -verify -pedantic -fsyntax-only

__constant int ci = 1;

// __constant ints are allowed in constant expressions.
enum use_ci_in_enum { enumerator = ci };
typedef int use_ci_in_array_bound[ci];

// general constant folding of array bounds is not permitted
typedef int folding_in_array_bounds[&ci + 3 - &ci]; // expected-error-re {{{{variable length arrays are not supported in OpenCL|array size is not a constant expression}}}} expected-note {{cannot refer to element 3}}

__kernel void foo(__global int *gip) {
  __local int li;
  __local int lj = 2; // expected-error {{'__local' variable cannot have an initializer}}

  int *ip;
#if ((!__OPENCL_CPP_VERSION__) && (__OPENCL_C_VERSION__ < 200))
  ip = gip; // expected-error {{assigning '__global int *__private' to '__private int *__private' changes address space of pointer}}
  ip = &li; // expected-error {{assigning '__local int *' to '__private int *__private' changes address space of pointer}}
  ip = &ci; // expected-error {{assigning '__constant int *' to '__private int *__private' changes address space of pointer}}
#else
  ip = gip;
  ip = &li;
  ip = &ci;
#if !__OPENCL_CPP_VERSION__
// expected-error@-2 {{assigning '__constant int *' to '__generic int *__private' changes address space of pointer}}
#else
// expected-error@-4 {{assigning '__constant int *' to '__generic int *' changes address space of pointer}}
#endif
#endif
}

void explicit_cast(__global int *g, __local int *l, __constant int *c, __private int *p, const __constant int *cc) {
  g = (__global int *)l;
#if !__OPENCL_CPP_VERSION__
// expected-error@-2 {{casting '__local int *' to type '__global int *' changes address space of pointer}}
#else
// expected-error@-4 {{C-style cast from '__local int *' to '__global int *' converts between mismatching address spaces}}
#endif
  g = (__global int *)c;
#if !__OPENCL_CPP_VERSION__
// expected-error@-2 {{casting '__constant int *' to type '__global int *' changes address space of pointer}}
#else
// expected-error@-4 {{C-style cast from '__constant int *' to '__global int *' converts between mismatching address spaces}}
#endif
  g = (__global int *)cc;
#if !__OPENCL_CPP_VERSION__
// expected-error@-2 {{casting 'const __constant int *' to type '__global int *' changes address space of pointer}}
#else
// expected-error@-4 {{C-style cast from 'const __constant int *' to '__global int *' converts between mismatching address spaces}}
#endif
  g = (__global int *)p;
#if !__OPENCL_CPP_VERSION__
// expected-error@-2 {{casting '__private int *' to type '__global int *' changes address space of pointer}}
#else
// expected-error@-4 {{C-style cast from '__private int *' to '__global int *' converts between mismatching address spaces}}
#endif
  l = (__local int *)g;
#if !__OPENCL_CPP_VERSION__
// expected-error@-2 {{casting '__global int *' to type '__local int *' changes address space of pointer}}
#else
// expected-error@-4 {{C-style cast from '__global int *' to '__local int *' converts between mismatching address spaces}}
#endif
  l = (__local int *)c;
#if !__OPENCL_CPP_VERSION__
// expected-error@-2 {{casting '__constant int *' to type '__local int *' changes address space of pointer}}
#else
// expected-error@-4 {{C-style cast from '__constant int *' to '__local int *' converts between mismatching address spaces}}
#endif
  l = (__local int *)cc;
#if !__OPENCL_CPP_VERSION__
// expected-error@-2 {{casting 'const __constant int *' to type '__local int *' changes address space of pointer}}
#else
// expected-error@-4 {{C-style cast from 'const __constant int *' to '__local int *' converts between mismatching address spaces}}
#endif
  l = (__local int *)p;
#if !__OPENCL_CPP_VERSION__
// expected-error@-2 {{casting '__private int *' to type '__local int *' changes address space of pointer}}
#else
// expected-error@-4 {{C-style cast from '__private int *' to '__local int *' converts between mismatching address spaces}}
#endif
  c = (__constant int *)g;
#if !__OPENCL_CPP_VERSION__
// expected-error@-2 {{casting '__global int *' to type '__constant int *' changes address space of pointer}}
#else
// expected-error@-4 {{C-style cast from '__global int *' to '__constant int *' converts between mismatching address spaces}}
#endif
  c = (__constant int *)l;
#if !__OPENCL_CPP_VERSION__
// expected-error@-2 {{casting '__local int *' to type '__constant int *' changes address space of pointer}}
#else
// expected-error@-4 {{C-style cast from '__local int *' to '__constant int *' converts between mismatching address spaces}}
#endif
  c = (__constant int *)p;
#if !__OPENCL_CPP_VERSION__
// expected-error@-2 {{casting '__private int *' to type '__constant int *' changes address space of pointer}}
#else
// expected-error@-4 {{C-style cast from '__private int *' to '__constant int *' converts between mismatching address spaces}}
#endif
  p = (__private int *)g;
#if !__OPENCL_CPP_VERSION__
// expected-error@-2 {{casting '__global int *' to type '__private int *' changes address space of pointer}}
#else
// expected-error@-4 {{C-style cast from '__global int *' to '__private int *' converts between mismatching address spaces}}
#endif
  p = (__private int *)l;
#if !__OPENCL_CPP_VERSION__
// expected-error@-2 {{casting '__local int *' to type '__private int *' changes address space of pointer}}
#else
// expected-error@-4 {{C-style cast from '__local int *' to '__private int *' converts between mismatching address spaces}}
#endif
  p = (__private int *)c;
#if !__OPENCL_CPP_VERSION__
// expected-error@-2 {{casting '__constant int *' to type '__private int *' changes address space of pointer}}
#else
// expected-error@-4 {{C-style cast from '__constant int *' to '__private int *' converts between mismatching address spaces}}
#endif
  p = (__private int *)cc;
#if !__OPENCL_CPP_VERSION__
// expected-error@-2 {{casting 'const __constant int *' to type '__private int *' changes address space of pointer}}
#else
// expected-error@-4 {{C-style cast from 'const __constant int *' to '__private int *' converts between mismatching address spaces}}
#endif
}

void ok_explicit_casts(__global int *g, __global int *g2, __local int *l, __local int *l2, __private int *p, __private int *p2) {
  g = (__global int *)g2;
  l = (__local int *)l2;
  p = (__private int *)p2;
}

#if !__OPENCL_CPP_VERSION__
void nested(__global int *g, __global int * __private *gg, __local int *l, __local int * __private *ll, __global float * __private *gg_f) {
  g = gg;    // expected-error {{assigning '__global int *__private *__private' to '__global int *__private' changes address space of pointer}}
  g = l;     // expected-error {{assigning '__local int *__private' to '__global int *__private' changes address space of pointer}}
  g = ll;    // expected-error {{assigning '__local int *__private *__private' to '__global int *__private' changes address space of pointer}}
  g = gg_f;  // expected-error {{assigning '__global float *__private *__private' to '__global int *__private' changes address space of pointer}}
  g = (__global int *)gg_f; // expected-error {{casting '__global float *__private *' to type '__global int *' changes address space of pointer}}

  gg = g;    // expected-error {{assigning '__global int *__private' to '__global int *__private *__private' changes address space of pointer}}
  gg = l;    // expected-error {{assigning '__local int *__private' to '__global int *__private *__private' changes address space of pointer}}
  gg = ll;   // expected-error {{assigning '__local int *__private *__private' to '__global int *__private *__private' changes address space of nested pointer}}
  gg = gg_f; // expected-warning {{incompatible pointer types assigning to '__global int *__private *__private' from '__global float *__private *__private'}}
  gg = (__global int * __private *)gg_f;

  l = g;     // expected-error {{assigning '__global int *__private' to '__local int *__private' changes address space of pointer}}
  l = gg;    // expected-error {{assigning '__global int *__private *__private' to '__local int *__private' changes address space of pointer}}
  l = ll;    // expected-error {{assigning '__local int *__private *__private' to '__local int *__private' changes address space of pointer}}
  l = gg_f;  // expected-error {{assigning '__global float *__private *__private' to '__local int *__private' changes address space of pointer}}
  l = (__local int *)gg_f; // expected-error {{casting '__global float *__private *' to type '__local int *' changes address space of pointer}}

  ll = g;    // expected-error {{assigning '__global int *__private' to '__local int *__private *__private' changes address space of pointer}}
  ll = gg;   // expected-error {{assigning '__global int *__private *__private' to '__local int *__private *__private' changes address space of nested pointer}}
  ll = l;    // expected-error {{assigning '__local int *__private' to '__local int *__private *__private' changes address space of pointer}}
  ll = gg_f; // expected-error {{assigning '__global float *__private *__private' to '__local int *__private *__private' changes address space of nested pointer}}
  ll = (__local int * __private *)gg_f; // expected-warning {{casting '__global float *__private *' to type '__local int *__private *' discards qualifiers in nested pointer types}}

  gg_f = g;  // expected-error {{assigning '__global int *__private' to '__global float *__private *__private' changes address space of pointer}}
  gg_f = gg; // expected-warning {{incompatible pointer types assigning to '__global float *__private *__private' from '__global int *__private *__private'}}
  gg_f = l;  // expected-error {{assigning '__local int *__private' to '__global float *__private *__private' changes address space of pointer}}
  gg_f = ll; // expected-error {{assigning '__local int *__private *__private' to '__global float *__private *__private' changes address space of nested pointer}}
  gg_f = (__global float * __private *)gg;

  // FIXME: This doesn't seem right. This should be an error, not a warning.
  __local int * __global * __private * lll;
  lll = gg; // expected-warning {{incompatible pointer types assigning to '__local int *__global *__private *__private' from '__global int *__private *__private'}}

  typedef __local int * l_t;
  typedef __global int * g_t;
  __private l_t * pl;
  __private g_t * pg;
  gg = pl;  // expected-error {{assigning '__private l_t *__private' (aka '__local int *__private *__private') to '__global int *__private *__private' changes address space of nested pointer}}
  pl = gg;  // expected-error {{assigning '__global int *__private *__private' to '__private l_t *__private' (aka '__local int *__private *__private') changes address space of nested pointer}}
  gg = pg;
  pg = gg;
  pg = pl;  // expected-error {{assigning '__private l_t *__private' (aka '__local int *__private *__private') to '__private g_t *__private' (aka '__global int *__private *__private') changes address space of nested pointer}}
  pl = pg;  // expected-error {{assigning '__private g_t *__private' (aka '__global int *__private *__private') to '__private l_t *__private' (aka '__local int *__private *__private') changes address space of nested pointer}}

  ll = (__local int * __private *)(void *)gg;
  void *vp = ll;
}
#else
void nested(__global int *g, __global int * __private *gg, __local int *l, __local int * __private *ll, __global float * __private *gg_f) {
  g = gg;    // expected-error {{assigning '__global int *__private *__private' to '__global int *' changes address space of pointer}}
  g = l;     // expected-error {{assigning '__local int *__private' to '__global int *' changes address space of pointer}}
  g = ll;    // expected-error {{assigning '__local int *__private *__private' to '__global int *' changes address space of pointer}}
  g = gg_f;  // expected-error {{assigning '__global float *__private *__private' to '__global int *' changes address space of pointer}}
  g = (__global int *)gg_f; // expected-error {{C-style cast from '__global float *__private *' to '__global int *' converts between mismatching address spaces}}

  gg = g;    // expected-error {{assigning '__global int *__private' to '__global int *__private *' changes address space of pointer}}
  gg = l;    // expected-error {{assigning '__local int *__private' to '__global int *__private *' changes address space of pointer}}
  gg = ll;   // expected-error {{assigning '__local int *__private *__private' to '__global int *__private *' changes address space of nested pointer}}
  gg = gg_f; // expected-error {{incompatible pointer types assigning to '__global int *__private *' from '__global float *__private *__private'}}
  gg = (__global int * __private *)gg_f;

  l = g;     // expected-error {{assigning '__global int *__private' to '__local int *' changes address space of pointer}}
  l = gg;    // expected-error {{assigning '__global int *__private *__private' to '__local int *' changes address space of pointer}}
  l = ll;    // expected-error {{assigning '__local int *__private *__private' to '__local int *' changes address space of pointer}}
  l = gg_f;  // expected-error {{assigning '__global float *__private *__private' to '__local int *' changes address space of pointer}}
  l = (__local int *)gg_f; // expected-error {{C-style cast from '__global float *__private *' to '__local int *' converts between mismatching address spaces}}

  ll = g;    // expected-error {{assigning '__global int *__private' to '__local int *__private *' changes address space of pointer}}
  ll = gg;   // expected-error {{assigning '__global int *__private *__private' to '__local int *__private *' changes address space of nested pointer}}
  ll = l;    // expected-error {{assigning '__local int *__private' to '__local int *__private *' changes address space of pointer}}
  ll = gg_f; // expected-error {{assigning '__global float *__private *__private' to '__local int *__private *' changes address space of nested pointer}}
  ll = (__local int *__private *)gg; //expected-warning{{C-style cast from '__global int *__private *' to '__local int *__private *' changes address space of nested pointers}}

  gg_f = g;  // expected-error {{assigning '__global int *__private' to '__global float *__private *' changes address space of pointer}}
  gg_f = gg; // expected-error {{incompatible pointer types assigning to '__global float *__private *' from '__global int *__private *__private'}}
  gg_f = l;  // expected-error {{assigning '__local int *__private' to '__global float *__private *' changes address space of pointer}}
  gg_f = ll; // expected-error {{assigning '__local int *__private *__private' to '__global float *__private *' changes address space of nested pointer}}
  gg_f = (__global float * __private *)gg;

  typedef __local int * l_t;
  typedef __global int * g_t;
  __private l_t * pl;
  __private g_t * pg;
  gg = pl;  // expected-error {{assigning '__private l_t *__private' (aka '__local int *__private *__private') to '__global int *__private *' changes address space of nested pointer}}
  pl = gg;  // expected-error {{assigning '__global int *__private *__private' to '__private l_t *' (aka '__local int *__private *') changes address space of nested pointer}}
  gg = pg;
  pg = gg;
  pg = pl;  // expected-error {{assigning '__private l_t *__private' (aka '__local int *__private *__private') to '__private g_t *' (aka '__global int *__private *') changes address space of nested pointer}}
  pl = pg;  // expected-error {{assigning '__private g_t *__private' (aka '__global int *__private *__private') to '__private l_t *' (aka '__local int *__private *') changes address space of nested pointer}}

  ll = (__local int * __private *)(void *)gg;
  void *vp = ll;
}
#endif

__private int func_return_priv(void);       //expected-error {{return type cannot be qualified with address space}}
__global int func_return_global(void);      //expected-error {{return type cannot be qualified with address space}}
__local int func_return_local(void);        //expected-error {{return type cannot be qualified with address space}}
__constant int func_return_constant(void);  //expected-error {{return type cannot be qualified with address space}}
#if __OPENCL_C_VERSION__ >= 200
__generic int func_return_generic(void);    //expected-error {{return type cannot be qualified with address space}}
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

void func_with_array_param(const unsigned data[16]);

__kernel void k() {
  unsigned data[16];
  func_with_array_param(data);
}

void func_multiple_addr2(void) {
  typedef __private int private_int_t;
  __attribute__((opencl_global)) __private int var1;   // expected-error {{multiple address spaces specified for type}} \
                                                       // expected-error {{function scope variable cannot be declared in global address space}}
  __private __attribute__((opencl_global)) int *var2;  // expected-error {{multiple address spaces specified for type}}
  __attribute__((opencl_global)) private_int_t var3;   // expected-error {{multiple address spaces specified for type}}
  __attribute__((opencl_global)) private_int_t *var4;  // expected-error {{multiple address spaces specified for type}}
  __attribute__((opencl_private)) private_int_t var5;  // expected-warning {{multiple identical address spaces specified for type}}
  __attribute__((opencl_private)) private_int_t *var6; // expected-warning {{multiple identical address spaces specified for type}}
#if __OPENCL_CPP_VERSION__
  __global int [[clang::opencl_private]] var7;         // expected-error {{multiple address spaces specified for type}}
  __global int [[clang::opencl_private]] *var8;        // expected-error {{multiple address spaces specified for type}}
  private_int_t [[clang::opencl_private]] var9;        // expected-warning {{multiple identical address spaces specified for type}}
  private_int_t [[clang::opencl_private]] *var10;      // expected-warning {{multiple identical address spaces specified for type}}
#endif // !__OPENCL_CPP_VERSION__
}
