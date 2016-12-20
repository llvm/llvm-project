// RUN: %clang_cc1 -verify %s

struct {
  int a : 1; // expected-error {{bit-fields are not supported in OpenCL}}
};

void no_vla(int n) {
  int a[n]; // expected-error {{variable length arrays are not supported in OpenCL}}
}

void no_logxor(int n) {
  int logxor = n ^^ n; // expected-error {{^^ is a reserved operator in OpenCL}}
}
