// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=CL1.2
// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=CL3.0 -cl-ext=-all
// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=clc++2021 -cl-ext=-all

void foo(read_only pipe int p);
#if __OPENCL_C_VERSION__ > 120
// expected-error@-2 {{OpenCL C version 3.0 does not support the 'pipe' type qualifier}}
// expected-error@-3 {{access qualifier can only be used for pipe and image type}}
#elif defined(__OPENCL_CPP_VERSION__)
// expected-error@-5 {{C++ for OpenCL version 2021 does not support the 'pipe' type qualifier}}
// expected-error@-6 {{access qualifier can only be used for pipe and image type}}
#else
// expected-error@-8 {{type specifier missing, defaults to 'int'}}
// expected-error@-9 {{access qualifier can only be used for pipe and image type}}
// expected-error@-10 {{expected ')'}} expected-note@-10 {{to match this '('}}
#endif

// 'pipe' should be accepted as an identifier.
typedef int pipe;
#if __OPENCL_C_VERSION__ > 120
// expected-error@-2 {{OpenCL C version 3.0 does not support the 'pipe' type qualifier}}
// expected-warning@-3 {{typedef requires a name}}
#elif defined(__OPENCL_CPP_VERSION__)
// expected-error@-5 {{C++ for OpenCL version 2021 does not support the 'pipe' type qualifier}}
// expected-warning@-6 {{typedef requires a name}}
#endif

void bar(void) {
 reserve_id_t r;
#if defined(__OPENCL_C_VERSION__)
// expected-error@-2 {{use of undeclared identifier 'reserve_id_t'}}
#else
// expected-error@-4 {{unknown type name 'reserve_id_t'}}
#endif
}
