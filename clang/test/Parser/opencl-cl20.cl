// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only
// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=CL2.0 -DCL20

#ifdef CL20
// expected-no-diagnostics
#endif

__generic int * __generic_test(__generic int *arg) {
  __generic int *var;
  return var;  
}
#ifndef CL20
// expected-error@-5 {{OpenCL C version 1.0 does not support the '__generic' type qualifier}}
// expected-error@-6 {{OpenCL C version 1.0 does not support the '__generic' type qualifier}}
// expected-error@-6 {{OpenCL C version 1.0 does not support the '__generic' type qualifier}}
#endif

generic int * generic_test(generic int *arg) {
  generic int *var;
  return var;  
}
#ifndef CL20
// expected-error@-5 {{OpenCL C version 1.0 does not support the 'generic' type qualifier}}
// expected-error@-6 {{OpenCL C version 1.0 does not support the 'generic' type qualifier}}
// expected-error@-6 {{OpenCL C version 1.0 does not support the 'generic' type qualifier}}
#endif
