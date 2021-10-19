// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=CL2.0
// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=clc++1.0

int G2 = 0;
global int G3 = 0;
local int G4 = 0;              // expected-error{{program scope variable must reside in global or constant address space}}

static float g_implicit_static_var = 0;
static constant float g_constant_static_var = 0;
static global float g_global_static_var = 0;
static local float g_local_static_var = 0;     // expected-error {{program scope variable must reside in global or constant address space}}
static private float g_private_static_var = 0; // expected-error {{program scope variable must reside in global or constant address space}}
static generic float g_generic_static_var = 0; // expected-error {{program scope variable must reside in global or constant address space}}

extern float g_implicit_extern_var;
extern constant float g_constant_extern_var;
extern global float g_global_extern_var;
extern local float g_local_extern_var;     // expected-error {{extern variable must reside in global or constant address space}}
extern private float g_private_extern_var; // expected-error {{extern variable must reside in global or constant address space}}
extern generic float g_generic_extern_var; // expected-error {{extern variable must reside in global or constant address space}}

static void kernel bar() { // expected-error{{kernel functions cannot be declared static}}
}

void kernel foo() {
  constant int L1 = 0;
  local int L2;
  global int L3;                              // expected-error{{function scope variable cannot be declared in global address space}}
  generic int L4;                             // expected-error{{automatic variable qualified with an invalid address space}}
  __attribute__((address_space(100))) int L5; // expected-error{{automatic variable qualified with an invalid address space}}

  static float l_implicit_static_var = 0;
  static constant float l_constant_static_var = 0;
  static global float l_global_static_var = 0;
  static local float l_local_static_var = 0;     // expected-error {{static local variable must reside in global or constant address space}}
  static private float l_private_static_var = 0; // expected-error {{static local variable must reside in global or constant address space}}
  static generic float l_generic_static_var = 0; // expected-error {{static local variable must reside in global or constant address space}}

  extern float l_implicit_extern_var;
  extern constant float l_constant_extern_var;
  extern global float l_global_extern_var;
  extern local float l_local_extern_var;     // expected-error {{extern variable must reside in global or constant address space}}
  extern private float l_private_extern_var; // expected-error {{extern variable must reside in global or constant address space}}
  extern generic float l_generic_extern_var; // expected-error {{extern variable must reside in global or constant address space}}
}
