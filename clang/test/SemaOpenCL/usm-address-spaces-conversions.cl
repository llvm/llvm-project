// RUN: %clang_cc1 %s -ffake-address-space-map -verify -pedantic -fsyntax-only -cl-std=CL2.0
// RUN: %clang_cc1 %s -ffake-address-space-map -verify -pedantic -fsyntax-only -cl-std=CL2.0 -DGENERIC
// RUN: %clang_cc1 %s -ffake-address-space-map -verify -pedantic -fsyntax-only -cl-std=CL2.0 -DCONSTANT
// RUN: %clang_cc1 %s -ffake-address-space-map -verify -pedantic -fsyntax-only -cl-std=CL2.0 -DLOCAL

/* USM (unified shared memory) extension for OpenCLC 2.0 adds two new address
 * spaces: global_device and global_host that are a subset of __global address
 * space. As ISO/IEC TR 18037 5.1.3 declares - it's possible to implicitly
 * convert a subset address space to a superset address space, while conversion
 * in a reversed direction could be achived only with an explicit cast */

#ifdef GENERIC
#define AS_COMP __generic
#else
#define AS_COMP __global
#endif // GENERIC

#ifdef CONSTANT
#define AS_INCOMP __constant
#elif LOCAL
#define AS_INCOMP __local
#else // PRIVATE
#define AS_INCOMP __private
#endif // CONSTANT

void test(AS_COMP int *arg_comp,
          __attribute__((opencl_global_device)) int *arg_device,
          __attribute__((opencl_global_host)) int *arg_host) {
  AS_COMP int *var_glob1 = arg_device;
  AS_COMP int *var_glob2 = arg_host;
  AS_COMP int *var_glob3 = (AS_COMP int *)arg_device;
  AS_COMP int *var_glob4 = (AS_COMP int *)arg_host;
  arg_device = (__attribute__((opencl_global_device)) int *)arg_comp;
  arg_host = (__attribute__((opencl_global_host)) int *)arg_comp;
#ifdef GENERIC
  // expected-error@+6{{assigning '__generic int *__private' to '__global_device int *__private' changes address space of pointer}}
  // expected-error@+6{{assigning '__generic int *__private' to '__global_host int *__private' changes address space of pointer}}
#else
  // expected-error@+3{{assigning '__global int *__private' to '__global_device int *__private' changes address space of pointer}}
  // expected-error@+3{{assigning '__global int *__private' to '__global_host int *__private' changes address space of pointer}}
#endif // GENERIC
  arg_device = arg_comp;
  arg_host = arg_comp;

#ifdef CONSTANT
  // expected-error@+15{{initializing '__constant int *__private' with an expression of type '__global_device int *__private' changes address space of pointer}}
  // expected-error@+15{{initializing '__constant int *__private' with an expression of type '__global_host int *__private' changes address space of pointer}}
  // expected-error@+15{{initializing '__constant int *__private' with an expression of type '__global_device int *' changes address space of pointer}}
  // expected-error@+16{{initializing '__constant int *__private' with an expression of type '__global_host int *' changes address space of pointer}}
#elif LOCAL
  // expected-error@+10{{initializing '__local int *__private' with an expression of type '__global_device int *__private' changes address space of pointer}}
  // expected-error@+10{{initializing '__local int *__private' with an expression of type '__global_host int *__private' changes address space of pointer}}
  // expected-error@+10{{initializing '__local int *__private' with an expression of type '__global_device int *' changes address space of pointer}}
  // expected-error@+11{{initializing '__local int *__private' with an expression of type '__global_host int *' changes address space of pointer}}
#else // PRIVATE
  // expected-error@+5{{initializing '__private int *__private' with an expression of type '__global_device int *__private' changes address space of pointer}}
  // expected-error@+5{{initializing '__private int *__private' with an expression of type '__global_host int *__private' changes address space of pointer}}
  // expected-error@+5{{initializing '__private int *__private' with an expression of type '__global_device int *' changes address space of pointer}}
  // expected-error@+6{{initializing '__private int *__private' with an expression of type '__global_host int *' changes address space of pointer}}
#endif // CONSTANT
  AS_INCOMP int *var_incomp1 = arg_device;
  AS_INCOMP int *var_incomp2 = arg_host;
  AS_INCOMP int *var_incomp3 =
      (__attribute__((opencl_global_device)) int *)arg_device;
  AS_INCOMP int *var_incomp4 =
      (__attribute__((opencl_global_host)) int *)arg_host;
}
