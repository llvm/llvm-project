// RUN: %clang_cc1 -verify -fsyntax-only %s
// RUN: %clang_cc1 -verify -fsyntax-only -cl-std=CL2.0 %s

void test(void) {
  global int *glob;
  local int *loc;
  constant int *con;
  typedef constant int const_int_ty;
  const_int_ty *con_typedef;

  glob = to_global(glob, loc);
#if __OPENCL_C_VERSION__ < CL_VERSION_2_0
  // expected-error@-2{{implicit declaration of function 'to_global' is invalid in OpenCL}}
  // expected-warning@-3{{incompatible integer to pointer conversion assigning to '__global int *' from 'int'}}
#else
  // expected-error@-5{{invalid number of arguments to function: 'to_global'}}
#endif

  int x;
  glob = to_global(x);
#if __OPENCL_C_VERSION__ < CL_VERSION_2_0
  // expected-warning@-2{{incompatible integer to pointer conversion assigning to '__global int *' from 'int'}}
#else
  // expected-error@-4{{invalid argument x to function: 'to_global', expecting a generic pointer argument}}
#endif

  glob = to_global(con);
#if __OPENCL_C_VERSION__ < CL_VERSION_2_0
  // expected-warning@-2{{incompatible integer to pointer conversion assigning to '__global int *' from 'int'}}
#else
  // expected-error@-4{{invalid argument con to function: 'to_global', expecting a generic pointer argument}}
#endif

  glob = to_global(con_typedef);
#if __OPENCL_C_VERSION__ < CL_VERSION_2_0
  // expected-warning@-2{{incompatible integer to pointer conversion assigning to '__global int *' from 'int'}}
#else
  // expected-error@-4{{invalid argument con_typedef to function: 'to_global', expecting a generic pointer argument}}
#endif

  loc = to_global(glob);
#if __OPENCL_C_VERSION__ < CL_VERSION_2_0
  // expected-warning@-2{{incompatible integer to pointer conversion assigning to '__local int *' from 'int'}}
#else
  // expected-error@-4{{assigning '__global int *' to '__local int *' changes address space of pointer}}
#endif

  global char *glob_c = to_global(loc);
#if __OPENCL_C_VERSION__ < CL_VERSION_2_0
  // expected-warning@-2{{incompatible integer to pointer conversion initializing '__global char *' with an expression of type 'int'}}
#else
  // expected-warning@-4{{incompatible pointer types initializing '__global char *' with an expression of type '__global int *'}}
#endif

}
