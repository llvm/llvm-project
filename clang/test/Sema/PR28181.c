// RUN: %clang_cc1 -fsyntax-only -verify %s

struct spinlock_t {
  int lock;
} audit_skb_queue;

void fn1(void) {
  audit_skb_queue = (lock); // expected-error {{use of undeclared identifier 'lock'; did you mean 'long'?}}
}                           // expected-error@-1 {{assigning to 'struct spinlock_t' from incompatible type '<overloaded function type>'}}

void fn2(void) {
  audit_skb_queue + (lock); // expected-error {{use of undeclared identifier 'lock'; did you mean 'long'?}}
}                           // expected-error@-1 {{reference to overloaded function could not be resolved; did you mean to call it?}}
