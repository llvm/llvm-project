// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=CL1.2
// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=CL3.0 -cl-ext=-__opencl_c_pipes,-__opencl_c_generic_address_space

void foo(read_only pipe int p);
#if __OPENCL_C_VERSION__ > 120
// expected-error@-2 {{OpenCL C version 3.0 does not support the 'pipe' type qualifier}}
// expected-error@-3 {{access qualifier can only be used for pipe and image type}}
#else
// expected-warning@-5 {{type specifier missing, defaults to 'int'}}
// expected-error@-6 {{access qualifier can only be used for pipe and image type}}
// expected-error@-7 {{expected ')'}} expected-note@-7 {{to match this '('}}
#endif

// 'pipe' should be accepted as an identifier.
typedef int pipe;
#if __OPENCL_C_VERSION__ > 120
// expected-error@-2 {{OpenCL C version 3.0 does not support the 'pipe' type qualifier}}
// expected-warning@-3 {{typedef requires a name}}
#endif

void bar() {
 reserve_id_t r; // expected-error {{use of undeclared identifier 'reserve_id_t'}}
}
