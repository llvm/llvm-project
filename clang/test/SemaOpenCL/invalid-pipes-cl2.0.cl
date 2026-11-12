// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -Wno-strict-prototypes -cl-std=CL2.0
// RUN: %clang_cc1 %s -verify=expected,blocks -pedantic -fsyntax-only -Wno-strict-prototypes -cl-std=CL2.0 -fblocks -DTEST_BLOCK_POINTER_PACKET
// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -Wno-strict-prototypes -cl-std=CL3.0 -cl-ext=+__opencl_c_pipes,+__opencl_c_generic_address_space,+__opencl_c_program_scope_global_variables
// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -Wno-strict-prototypes -cl-std=CL3.0 -cl-ext=+__opencl_c_pipes,+__opencl_c_generic_address_space,-__opencl_c_program_scope_global_variables,-__opencl_c_device_enqueue
// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=clc++1.0
// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=clc++2021 -cl-ext=+__opencl_c_pipes,+__opencl_c_generic_address_space,+__opencl_c_program_scope_global_variables
// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=clc++2021 -cl-ext=+__opencl_c_pipes,+__opencl_c_generic_address_space,-__opencl_c_program_scope_global_variables,-__opencl_c_device_enqueue

global pipe int gp;            // expected-error {{type '__global read_only pipe int' can only be used as a function parameter in OpenCL}}
global reserve_id_t rid;          // expected-error {{the '__global reserve_id_t' type cannot be used to declare a program scope variable}}

extern pipe write_only int get_pipe(void); // expected-error {{'write_only' attribute only applies to parameters and typedefs}} expected-error-re {{non-object type 'int ({{(void)?}})' cannot be used as a pipe packet type}}

global pipe notype1, notype2; // expected-error {{missing actual type specifier for pipe}}

kernel void test_invalid_reserved_id(reserve_id_t ID) { // expected-error {{'__private reserve_id_t' cannot be used as the type of a kernel parameter}}
}

void test1(pipe int *p) {// expected-error {{pointer type '__generic int *' cannot be used as a pipe packet type}}
}

#ifdef __OPENCL_CPP_VERSION__
void reference_packet_type(pipe int &p) {} // expected-error {{reference type '__generic int &' cannot be used as a pipe packet type}}
#endif

#ifdef TEST_BLOCK_POINTER_PACKET
typedef void (^block_t)(void);
void block_pointer_packet_type(pipe block_t p) {} // blocks-error-re {{block pointer type 'block_t'{{.*}} cannot be used as a pipe packet type}}
#endif

typedef pipe void invalid_pipe_typedef; // expected-error {{non-object type 'void' cannot be used as a pipe packet type}}

typedef void function_type(void);
void non_object_packet_types(
    pipe void p1, // expected-error {{non-object type 'void' cannot be used as a pipe packet type}}
    pipe function_type p2) {} // expected-error-re {{non-object type 'function_type' (aka 'void ({{(void)?}})') cannot be used as a pipe packet type}}

struct incomplete;
typedef pipe struct incomplete invalid_incomplete_pipe; // expected-error {{incomplete type 'struct incomplete' cannot be used as a pipe packet type}}

#ifdef __OPENCL_CPP_VERSION__
template <typename T>
void non_object_packet_in_template(pipe void p) {} // expected-error {{non-object type 'void' cannot be used as a pipe packet type}}

template <typename T>
void dependent_packet_type(pipe T p) {}

template <typename T> using dependent_function_type = T();
template <typename T>
void dependent_function_packet_type(pipe dependent_function_type<T> p) {} // expected-error {{non-object type 'dependent_function_type<T>' (aka 'T ()') cannot be used as a pipe packet type}}

template <typename T> struct packet_type { T value; };
void class_template_packet_type(pipe packet_type<int> p) {}
#endif

void test2(pipe p) {// expected-error {{missing actual type specifier for pipe}}
}
void test3(int pipe p) {// expected-error {{cannot combine with previous 'int' declaration specifier}}
}
void test4(void) {
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
pipe_int_t test6(void) {} // expected-error{{declaring function return value of type 'pipe_int_t' (aka 'read_only pipe int') is not allowed}}

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
