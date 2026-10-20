// RUN: %clang_cc1 -verify %s
// RUN: %clang_cc1 -verify %s -DBITFIELDS_EXT -triple spir

#ifdef BITFIELDS_EXT
#pragma OPENCL EXTENSION __cl_clang_bitfields : enable
#endif

struct test {
  int a : 1;
#ifndef BITFIELDS_EXT
// expected-error@-2 {{bit-fields are not supported in OpenCL}}
#endif
};

void no_vla(int n) {
  int a[n]; // expected-error {{variable length arrays are not supported in OpenCL}}
}

void no_logxor(int n) {
  int logxor = n ^^ n; // expected-error {{^^ is a reserved operator in OpenCL}} \
                          expected-error {{type name requires a specifier or qualifier}} \
                          expected-error {{expected expression}}
}
