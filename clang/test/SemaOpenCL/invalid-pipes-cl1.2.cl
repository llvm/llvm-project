// RUN: %clang_cc1 %s -verify=expected,cl12 -pedantic -fsyntax-only \
// RUN:   -cl-std=CL1.2
// RUN: %clang_cc1 %s -verify=expected,cl3 -pedantic -fsyntax-only \
// RUN:   -cl-std=CL3.0 -cl-ext=-all
// RUN: %clang_cc1 %s -verify=expected,clcpp -pedantic -fsyntax-only \
// RUN:   -cl-std=clc++2021 -cl-ext=-all

// cl3-error@+6 {{OpenCL C version 3.0 does not support the 'pipe' type qualifier}}
// clcpp-error@+5 {{C++ for OpenCL version 2021 does not support the 'pipe' type qualifier}}
// cl12-error@+4 {{type specifier missing, defaults to 'int'}}
// expected-error@+3 {{access qualifier can only be used for pipe and image type}}
// cl12-error@+2 {{expected ')'}}
// cl12-note@+1 {{to match this '('}}
void unavailable_pipe_parameter(read_only pipe int p);

// 'pipe' is accepted as an identifier in OpenCL 1.2.
// cl3-error@+4 {{OpenCL C version 3.0 does not support the 'pipe' type qualifier}}
// cl3-warning@+3 {{typedef requires a name}}
// clcpp-error@+2 {{C++ for OpenCL version 2021 does not support the 'pipe' type qualifier}}
// clcpp-warning@+1 {{typedef requires a name}}
typedef int pipe;

void unavailable_reserve_id_type(void) {
  // cl12-error@+3 {{use of undeclared identifier 'reserve_id_t'}}
  // cl3-error@+2 {{use of undeclared identifier 'reserve_id_t'}}
  // clcpp-error@+1 {{unknown type name 'reserve_id_t'}}
  reserve_id_t r;
}
