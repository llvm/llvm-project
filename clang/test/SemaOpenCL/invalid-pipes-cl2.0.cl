// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only \
// RUN:   -Wno-strict-prototypes -cl-std=CL2.0
// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only \
// RUN:   -Wno-strict-prototypes -cl-std=CL3.0 \
// RUN:   -cl-ext=+__opencl_c_pipes,+__opencl_c_generic_address_space,+__opencl_c_program_scope_global_variables
// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only \
// RUN:   -Wno-strict-prototypes -cl-std=CL3.0 \
// RUN:   -cl-ext=+__opencl_c_pipes,+__opencl_c_generic_address_space,-__opencl_c_program_scope_global_variables,-__opencl_c_device_enqueue
// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only \
// RUN:   -cl-std=clc++1.0
// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only \
// RUN:   -cl-std=clc++2021 \
// RUN:   -cl-ext=+__opencl_c_pipes,+__opencl_c_generic_address_space,+__opencl_c_program_scope_global_variables
// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only \
// RUN:   -cl-std=clc++2021 \
// RUN:   -cl-ext=+__opencl_c_pipes,+__opencl_c_generic_address_space,-__opencl_c_program_scope_global_variables,-__opencl_c_device_enqueue

// expected-error@+1 {{type '__global read_only pipe int' can only be used as a function parameter in OpenCL}}
global pipe int gp;
// expected-error@+1 {{the '__global reserve_id_t' type cannot be used to declare a program scope variable}}
global reserve_id_t rid;

// expected-error@+1 {{'write_only' attribute only applies to parameters and typedefs}}
extern pipe write_only int get_pipe(void);
#if (__OPENCL_CPP_VERSION__ == 100) || (__OPENCL_C_VERSION__ == 200) || ((__OPENCL_CPP_VERSION__ == 202100 || __OPENCL_C_VERSION__ == 300) && defined(__opencl_c_program_scope_global_variables))
// expected-error-re@-2 {{type '__global write_only pipe int ({{(void)?}})' can only be used as a function parameter in OpenCL}}
#else
// FIXME: '__private' here makes no sense since program scope variables feature is not supported, should diagnose as '__global' probably
// expected-error-re@-5 {{type '__private write_only pipe int ({{(void)?}})' can only be used as a function parameter in OpenCL}}
#endif

// expected-error@+1 {{missing actual type specifier for pipe}}
global pipe notype1, notype2;

// expected-error@+1 {{'__private reserve_id_t' cannot be used as the type of a kernel parameter}}
kernel void invalid_reserved_id_parameter(reserve_id_t ID) {}

// expected-error@+1 {{pipes packet types cannot be of reference type}}
void pointer_packet_type(pipe int *p) {}
// expected-error@+1 {{missing actual type specifier for pipe}}
void missing_packet_type(pipe p) {}
// expected-error@+1 {{cannot combine with previous 'int' declaration specifier}}
void misplaced_pipe_specifier(int pipe p) {}

void local_pipe_variable(void) {
  // expected-error@+1 {{type '__private read_only pipe int' can only be used as a function parameter}}
  pipe int p;
  // TODO: Fix parsing of this pipe int (*p).
}

void invalid_pipe_operators(pipe int p) {
  // expected-error@+1 {{invalid operands to binary expression ('__private read_only pipe int' and '__private read_only pipe int')}}
  p + p;
  // expected-error@+1 {{invalid operands to binary expression ('__private read_only pipe int' and '__private read_only pipe int')}}
  p = p;
  // expected-error@+1 {{invalid argument type '__private read_only pipe int' to unary expression}}
  &p;
  // expected-error@+1 {{invalid argument type '__private read_only pipe int' to unary expression}}
  *p;
}

typedef pipe int pipe_int_t;
// expected-error@+1 {{declaring function return value of type 'pipe_int_t' (aka 'read_only pipe int') is not allowed}}
pipe_int_t pipe_return_type(void) {}

bool compare_reserve_ids(void) {
  reserve_id_t id1, id2;
  // expected-error@+1 {{invalid operands to binary expression ('__private reserve_id_t' and '__private reserve_id_t')}}
  return (id1 == id2);
}

// Pipe parameters with different packet types are incompatible.
#ifndef __OPENCL_CPP_VERSION__
// expected-note@+1 {{previous declaration is here}}
int merge_pipe_parameter_types(pipe int x, int y);
// expected-error@+1 {{conflicting types for 'merge_pipe_parameter_types'}}
int merge_pipe_parameter_types(x, y)
pipe short x;
int y;
{
  return y;
}
#endif
