// RUN: %clang_cc1 -verify -fsyntax-only %s
// RUN: %clang_cc1 -Wconversion -verify -fsyntax-only -cl-std=CL2.0 %s
// RUN: %clang_cc1 -Wconversion -verify -fsyntax-only -cl-std=CL3.0 -cl-ext=-all %s
// RUN: %clang_cc1 -Wconversion -verify -fsyntax-only -cl-std=CL3.0 %s

void test(void) {
  global int *glob;
  local int *loc;
  constant int *con;
  private int *priv;
  global float *glob_wrong_ty;
  typedef constant int const_int_ty;
  const_int_ty *con_typedef;

  glob = to_global(glob, loc);
#if (__OPENCL_C_VERSION__ < CL_VERSION_2_0) || (__OPENCL_C_VERSION__ >= CL_VERSION_3_0 && !defined(__opencl_c_generic_address_space))
  // expected-error@-2{{implicit declaration of function 'to_global' is invalid in OpenCL}}
  // expected-warning@-3{{incompatible integer to pointer conversion assigning to '__global int *__private' from 'int'}}
#else
  // expected-error@-5{{too many arguments to function call, expected 1, have 2}}
#endif

  int x;
  glob = to_global(x);
#if (__OPENCL_C_VERSION__ < CL_VERSION_2_0) || (__OPENCL_C_VERSION__ >= CL_VERSION_3_0 && !defined(__opencl_c_generic_address_space))
  // expected-warning@-2{{incompatible integer to pointer conversion assigning to '__global int *__private' from 'int'}}
#else
  // expected-error@-4{{invalid argument x to function: 'to_global', expecting a generic pointer argument}}
#endif

  glob = to_global(con);
#if (__OPENCL_C_VERSION__ < CL_VERSION_2_0) || (__OPENCL_C_VERSION__ >= CL_VERSION_3_0 && !defined(__opencl_c_generic_address_space))
  // expected-warning@-2{{incompatible integer to pointer conversion assigning to '__global int *__private' from 'int'}}
#else
  // expected-error@-4{{invalid argument con to function: 'to_global', expecting a generic pointer argument}}
#endif

  glob = to_global(con_typedef);
#if (__OPENCL_C_VERSION__ < CL_VERSION_2_0) || (__OPENCL_C_VERSION__ >= CL_VERSION_3_0 && !defined(__opencl_c_generic_address_space))
  // expected-warning@-2{{incompatible integer to pointer conversion assigning to '__global int *__private' from 'int'}}
#else
  // expected-error@-4{{invalid argument con_typedef to function: 'to_global', expecting a generic pointer argument}}
#endif

  loc = to_global(glob);
#if (__OPENCL_C_VERSION__ < CL_VERSION_2_0) || (__OPENCL_C_VERSION__ >= CL_VERSION_3_0 && !defined(__opencl_c_generic_address_space))
  // expected-warning@-2{{incompatible integer to pointer conversion assigning to '__local int *__private' from 'int'}}
#else
  // expected-error@-4{{assigning '__global int *' to '__local int *__private' changes address space of pointer}}
  // expected-warning@-5{{passing non-generic address space pointer to to_global may cause dynamic conversion affecting performance}}
#endif

  loc = to_private(glob);
#if (__OPENCL_C_VERSION__ < CL_VERSION_2_0) || (__OPENCL_C_VERSION__ >= CL_VERSION_3_0 && !defined(__opencl_c_generic_address_space))
  // expected-error@-2{{implicit declaration of function 'to_private' is invalid in OpenCL}}
  // expected-warning@-3{{incompatible integer to pointer conversion assigning to '__local int *__private' from 'int'}}
#else
  // expected-error@-5{{assigning '__private int *' to '__local int *__private' changes address space of pointer}}
  // expected-warning@-6{{passing non-generic address space pointer to to_private may cause dynamic conversion affecting performance}}
#endif

  loc = to_local(glob);
#if (__OPENCL_C_VERSION__ < CL_VERSION_2_0) || (__OPENCL_C_VERSION__ >= CL_VERSION_3_0 && !defined(__opencl_c_generic_address_space))
  // expected-error@-2{{implicit declaration of function 'to_local' is invalid in OpenCL}}
  // expected-warning@-3{{incompatible integer to pointer conversion assigning to '__local int *__private' from 'int'}}
  // expected-note@-4{{did you mean 'to_global'}}
  // expected-note@15{{'to_global' declared here}}
#else
  // expected-warning@-7{{passing non-generic address space pointer to to_local may cause dynamic conversion affecting performance}}
#endif

  priv = to_global(glob);
#if (__OPENCL_C_VERSION__ < CL_VERSION_2_0) || (__OPENCL_C_VERSION__ >= CL_VERSION_3_0 && !defined(__opencl_c_generic_address_space))
  // expected-warning@-2{{incompatible integer to pointer conversion assigning to '__private int *__private' from 'int'}}
#else
  // expected-error@-4{{assigning '__global int *' to '__private int *__private' changes address space of pointer}}
  // expected-warning@-5{{passing non-generic address space pointer to to_global may cause dynamic conversion affecting performance}}
#endif

  priv = to_private(glob);
#if (__OPENCL_C_VERSION__ < CL_VERSION_2_0) || (__OPENCL_C_VERSION__ >= CL_VERSION_3_0 && !defined(__opencl_c_generic_address_space))
  // expected-warning@-2{{incompatible integer to pointer conversion assigning to '__private int *__private' from 'int'}}
#else
  // expected-warning@-4{{passing non-generic address space pointer to to_private may cause dynamic conversion affecting performance}}
#endif


  priv = to_local(glob);
#if (__OPENCL_C_VERSION__ < CL_VERSION_2_0) || (__OPENCL_C_VERSION__ >= CL_VERSION_3_0 && !defined(__opencl_c_generic_address_space))
  // expected-warning@-2{{incompatible integer to pointer conversion assigning to '__private int *__private' from 'int'}}
#else
  // expected-error@-4{{assigning '__local int *' to '__private int *__private' changes address space of pointer}}
  // expected-warning@-5{{passing non-generic address space pointer to to_local may cause dynamic conversion affecting performance}}
#endif

  glob = to_global(glob);
#if (__OPENCL_C_VERSION__ < CL_VERSION_2_0) || (__OPENCL_C_VERSION__ >= CL_VERSION_3_0 && !defined(__opencl_c_generic_address_space))
  // expected-warning@-2{{incompatible integer to pointer conversion assigning to '__global int *__private' from 'int'}}
#else
  // expected-warning@-4{{passing non-generic address space pointer to to_global may cause dynamic conversion affecting performance}}
#endif

  glob = to_private(glob);
#if (__OPENCL_C_VERSION__ < CL_VERSION_2_0) || (__OPENCL_C_VERSION__ >= CL_VERSION_3_0 && !defined(__opencl_c_generic_address_space))
  // expected-warning@-2{{incompatible integer to pointer conversion assigning to '__global int *__private' from 'int'}}
#else
  // expected-error@-4{{assigning '__private int *' to '__global int *__private' changes address space of pointer}}
  // expected-warning@-5{{passing non-generic address space pointer to to_private may cause dynamic conversion affecting performance}}
#endif

  glob = to_local(glob);
#if (__OPENCL_C_VERSION__ < CL_VERSION_2_0) || (__OPENCL_C_VERSION__ >= CL_VERSION_3_0 && !defined(__opencl_c_generic_address_space))
  // expected-warning@-2{{incompatible integer to pointer conversion assigning to '__global int *__private' from 'int'}}
#else
  // expected-error@-4{{assigning '__local int *' to '__global int *__private' changes address space of pointer}}
  // expected-warning@-5{{passing non-generic address space pointer to to_local may cause dynamic conversion affecting performance}}
#endif

  global char *glob_c = to_global(loc);
#if (__OPENCL_C_VERSION__ < CL_VERSION_2_0) || (__OPENCL_C_VERSION__ >= CL_VERSION_3_0 && !defined(__opencl_c_generic_address_space))
  // expected-warning@-2{{incompatible integer to pointer conversion initializing '__global char *__private' with an expression of type 'int'}}
#else
  // expected-warning@-4{{incompatible pointer types initializing '__global char *__private' with an expression of type '__global int *'}}
  // expected-warning@-5{{passing non-generic address space pointer to to_global may cause dynamic conversion affecting performance}}
#endif

  glob_wrong_ty = to_global(glob);
#if (__OPENCL_C_VERSION__ < CL_VERSION_2_0) || (__OPENCL_C_VERSION__ >= CL_VERSION_3_0 && !defined(__opencl_c_generic_address_space))
  // expected-warning@-2{{incompatible integer to pointer conversion assigning to '__global float *__private' from 'int'}}
#else
  // expected-warning@-4{{incompatible pointer types assigning to '__global float *__private' from '__global int *'}}
  // expected-warning@-5{{passing non-generic address space pointer to to_global may cause dynamic conversion affecting performance}}
#endif

}
