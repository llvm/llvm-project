// RUN: %clang_cc1 %s -cl-std=CL2.0 -verify -pedantic -fsyntax-only
extern queue_t get_default_queue(void);

void queue_arg(queue_t); // expected-note {{passing argument to parameter here}}

void init(void) {
  queue_t q1 = 1; // expected-error{{initializing '__private queue_t' with an expression of incompatible type 'int'}}
  queue_t q = 0;
}

void assign(void) {
  queue_t q2, q3;
  q2 = 5; // expected-error{{assigning to '__private queue_t' from incompatible type 'int'}}
  q3 = 0;
  q2 = q3 = 0;
}

bool compare(void) {
  queue_t q4, q5;
  return 1 == get_default_queue() && // expected-error{{invalid operands to binary expression ('int' and 'queue_t')}}
         get_default_queue() == 1 && // expected-error{{invalid operands to binary expression ('queue_t' and 'int')}}
	     q4 == q5 &&
	     q4 != 0 &&
	     q4 != 0.0f; // expected-error{{invalid operands to binary expression ('__private queue_t' and 'float')}}
}

void call(void) {
  queue_arg(5); // expected-error {{passing 'int' to parameter of incompatible type 'queue_t'}}
  queue_arg(0);
}
