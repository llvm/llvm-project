// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=CL2.0

global pipe int gp;            // expected-error {{type '__global read_only pipe int' can only be used as a function parameter in OpenCL}}
global reserve_id_t rid;          // expected-error {{the '__global reserve_id_t' type cannot be used to declare a program scope variable}}

void test1(pipe int *p) {// expected-error {{pipes packet types cannot be of reference type}}
}
void test2(pipe p) {// expected-error {{missing actual type specifier for pipe}}
}
void test3(int pipe p) {// expected-error {{cannot combine with previous 'int' declaration specifier}}
}
void test4() {
  pipe int p; // expected-error {{type 'read_only pipe int' can only be used as a function parameter}}
  //TODO: fix parsing of this pipe int (*p);
}

void test5(pipe int p) {
  p+p; // expected-error{{invalid operands to binary expression ('read_only pipe int' and 'read_only pipe int')}}
  p=p; // expected-error{{invalid operands to binary expression ('read_only pipe int' and 'read_only pipe int')}}
  &p; // expected-error{{invalid argument type 'read_only pipe int' to unary expression}}
  *p; // expected-error{{invalid argument type 'read_only pipe int' to unary expression}}
}

typedef pipe int pipe_int_t;
pipe_int_t test6() {} // expected-error{{declaring function return value of type 'pipe_int_t' (aka 'read_only pipe int') is not allowed}}

bool test_id_comprision(void) {
  reserve_id_t id1, id2;
  return (id1 == id2);          // expected-error {{invalid operands to binary expression ('reserve_id_t' and 'reserve_id_t')}}
}

// Tests ASTContext::mergeTypes rejects this.
int f(pipe int x, int y); // expected-note {{previous declaration is here}}
int f(x, y) // expected-error {{conflicting types for 'f}}
pipe short x;
int y;
{
    return y;
}
