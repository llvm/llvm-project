// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=CL2.0

void test1(pipe int *p) {// expected-error {{pipes packet types cannot be of reference type}}
}
void test2(pipe p) {// expected-error {{missing actual type specifier for pipe}}
}
void test3(int pipe p) {// expected-error {{cannot combine with previous 'int' declaration specifier}}
}
void test4() {
  pipe int p; // expected-error {{type 'pipe int' can only be used as a function parameter}}
  //TODO: fix parsing of this pipe int (*p);
}

void test5(pipe int p) {
  p+p; // expected-error{{invalid operands to binary expression ('pipe int' and 'pipe int')}}
  p=p; // expected-error{{invalid operands to binary expression ('pipe int' and 'pipe int')}}
  &p; // expected-error{{invalid argument type 'pipe int' to unary expression}}
  *p; // expected-error{{invalid argument type 'pipe int' to unary expression}}
}

typedef pipe int pipe_int_t;
pipe_int_t test6() {} // expected-error{{declaring function return value of type 'pipe_int_t' (aka 'pipe int') is not allowed}}
