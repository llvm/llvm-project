// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=CL1.2

static constant int G1 = 0;
constant int G2 = 0;
int G3 = 0;        // expected-error{{program scope variable must reside in constant address space}}
global int G4 = 0; // expected-error{{program scope variable must reside in constant address space}}

static float g_implicit_static_var = 0; // expected-error {{program scope variable must reside in constant address space}}
static constant float g_constant_static_var = 0;
static global float g_global_static_var = 0;   // expected-error {{program scope variable must reside in constant address space}}
static local float g_local_static_var = 0;     // expected-error {{program scope variable must reside in constant address space}}
static private float g_private_static_var = 0; // expected-error {{program scope variable must reside in constant address space}}
static generic float g_generic_static_var = 0; // expected-error{{OpenCL C version 1.2 does not support the 'generic' type qualifier}} // expected-error {{program scope variable must reside in constant address space}}

extern float g_implicit_extern_var; // expected-error {{extern variable must reside in constant address space}}
extern constant float g_constant_extern_var;
extern global float g_global_extern_var;   // expected-error {{extern variable must reside in constant address space}}
extern local float g_local_extern_var;     // expected-error {{extern variable must reside in constant address space}}
extern private float g_private_extern_var; // expected-error {{extern variable must reside in constant address space}}
extern generic float g_generic_extern_var; // expected-error{{OpenCL C version 1.2 does not support the 'generic' type qualifier}} // expected-error {{extern variable must reside in constant address space}}

void kernel foo(int x) {
  // static is not allowed at local scope before CL2.0
  static int S1 = 5;          // expected-error{{variables in function scope cannot be declared static}}
  static constant int S2 = 5; // expected-error{{variables in function scope cannot be declared static}}

  constant int L1 = 0;
  local int L2;

  if (true) {
    local int L1; // expected-error {{variables in the local address space can only be declared in the outermost scope of a kernel function}}
    constant int L1 = 42; // expected-error {{variables in the constant address space can only be declared in the outermost scope of a kernel function}}
  }

  auto int L3 = 7;                            // expected-error{{OpenCL C version 1.2 does not support the 'auto' storage class specifier}}
  global int L4;                              // expected-error{{function scope variable cannot be declared in global address space}}
  __attribute__((address_space(100))) int L5; // expected-error{{automatic variable qualified with an invalid address space}}

  constant int L6 = x;                        // expected-error {{initializer element is not a compile-time constant}}
  global int *constant L7 = &G4;
  private int *constant L8 = &x;              // expected-error {{initializer element is not a compile-time constant}}
  constant int *constant L9 = &L1;
  local int *constant L10 = &L2;              // expected-error {{initializer element is not a compile-time constant}}
}

static void kernel bar() { // expected-error{{kernel functions cannot be declared static}}
}

void f() {
  constant int L1 = 0;                        // expected-error{{non-kernel function variable cannot be declared in constant address space}}
  local int L2;                               // expected-error{{non-kernel function variable cannot be declared in local address space}}
  global int L3;                              // expected-error{{function scope variable cannot be declared in global address space}}
  __attribute__((address_space(100))) int L4; // expected-error{{automatic variable qualified with an invalid address space}}

  {
    constant int L1 = 0;                        // expected-error{{non-kernel function variable cannot be declared in constant address space}}
    local int L2;                               // expected-error{{non-kernel function variable cannot be declared in local address space}}
    global int L3;                              // expected-error{{function scope variable cannot be declared in global address space}}
    __attribute__((address_space(100))) int L4; // expected-error{{automatic variable qualified with an invalid address space}}
  }

  static float l_implicit_static_var = 0;          // expected-error {{variables in function scope cannot be declared static}}
  static constant float l_constant_static_var = 0; // expected-error {{variables in function scope cannot be declared static}}
  static global float l_global_static_var = 0;     // expected-error {{variables in function scope cannot be declared static}}
  static local float l_local_static_var = 0;       // expected-error {{variables in function scope cannot be declared static}}
  static private float l_private_static_var = 0;   // expected-error {{variables in function scope cannot be declared static}}
  static generic float l_generic_static_var = 0;   // expected-error{{OpenCL C version 1.2 does not support the 'generic' type qualifier}} // expected-error {{variables in function scope cannot be declared static}}

  extern float l_implicit_extern_var; // expected-error {{extern variable must reside in constant address space}}
  extern constant float l_constant_extern_var;
  extern global float l_global_extern_var;   // expected-error {{extern variable must reside in constant address space}}
  extern local float l_local_extern_var;     // expected-error {{extern variable must reside in constant address space}}
  extern private float l_private_extern_var; // expected-error {{extern variable must reside in constant address space}}
  extern generic float l_generic_extern_var; // expected-error{{OpenCL C version 1.2 does not support the 'generic' type qualifier}} // expected-error {{extern variable must reside in constant address space}}
}
