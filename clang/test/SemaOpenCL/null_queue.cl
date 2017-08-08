// RUN: %clang_cc1 %s -cl-std=CL2.0 -verify -pedantic -fsyntax-only
extern queue_t get_default_queue();

bool compare() {
  return 1 == get_default_queue() && // expected-error{{invalid operands to binary expression ('int' and 'queue_t')}}
         get_default_queue() == 1; // expected-error{{invalid operands to binary expression ('queue_t' and 'int')}}
}

void init() {
  queue_t q1 = 1; // expected-error{{initializing 'queue_t' with an expression of incompatible type 'int'}}
  queue_t q = 0;
}
