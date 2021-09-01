// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=CL2.0
// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=CL3.0 -cl-ext=+__opencl_c_pipes,+__opencl_c_generic_address_space,+__opencl_c_program_scope_global_variables
// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=CL3.0 -cl-ext=+__opencl_c_pipes,+__opencl_c_generic_address_space,-__opencl_c_program_scope_global_variables
// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=clc++1.0
// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=clc++2021 -cl-ext=+__opencl_c_pipes,+__opencl_c_generic_address_space,+__opencl_c_program_scope_global_variables
// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=clc++2021 -cl-ext=+__opencl_c_pipes,+__opencl_c_generic_address_space,-__opencl_c_program_scope_global_variables

global pipe int gp;            // expected-error {{type '__global read_only pipe int' can only be used as a function parameter in OpenCL}}
global reserve_id_t rid;          // expected-error {{the '__global reserve_id_t' type cannot be used to declare a program scope variable}}

extern pipe write_only int get_pipe(); // expected-error {{'write_only' attribute only applies to parameters and typedefs}}
#if (__OPENCL_CPP_VERSION__ == 100) || (__OPENCL_C_VERSION__ == 200) || ((__OPENCL_CPP_VERSION__ == 202100 || __OPENCL_C_VERSION__ == 300) && defined(__opencl_c_program_scope_global_variables))
// expected-error-re@-2{{type '__global write_only pipe int ({{(void)?}})' can only be used as a function parameter in OpenCL}}
#else
// FIXME: '__private' here makes no sense since program scope variables feature is not supported, should diagnose as '__global' probably
// expected-error-re@-5{{type '__private write_only pipe int ({{(void)?}})' can only be used as a function parameter in OpenCL}}
#endif

kernel void test_invalid_reserved_id(reserve_id_t ID) { // expected-error {{'__private reserve_id_t' cannot be used as the type of a kernel parameter}}
}

void test1(pipe int *p) {// expected-error {{pipes packet types cannot be of reference type}}
}
void test2(pipe p) {// expected-error {{missing actual type specifier for pipe}}
}
void test3(int pipe p) {// expected-error {{cannot combine with previous 'int' declaration specifier}}
}
void test4() {
  pipe int p; // expected-error {{type '__private read_only pipe int' can only be used as a function parameter}}
  //TODO: fix parsing of this pipe int (*p);
}

void test5(pipe int p) {
  p+p; // expected-error{{invalid operands to binary expression ('__private read_only pipe int' and '__private read_only pipe int')}}
  p=p; // expected-error{{invalid operands to binary expression ('__private read_only pipe int' and '__private read_only pipe int')}}
  &p; // expected-error{{invalid argument type '__private read_only pipe int' to unary expression}}
  *p; // expected-error{{invalid argument type '__private read_only pipe int' to unary expression}}
}

typedef pipe int pipe_int_t;
pipe_int_t test6() {} // expected-error{{declaring function return value of type 'pipe_int_t' (aka 'read_only pipe int') is not allowed}}

bool test_id_comprision(void) {
  reserve_id_t id1, id2;
  return (id1 == id2);          // expected-error {{invalid operands to binary expression ('__private reserve_id_t' and '__private reserve_id_t')}}
}

// Tests ASTContext::mergeTypes rejects this.
#ifndef __OPENCL_CPP_VERSION__
int f(pipe int x, int y); // expected-note {{previous declaration is here}}
int f(x, y) // expected-error {{conflicting types for 'f}}
pipe short x;
int y;
{
    return y;
}
#endif
