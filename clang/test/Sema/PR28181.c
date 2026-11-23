// RUN: %clang_cc1 -fsyntax-only -verify %s

struct spinlock_t {
  int lock;
} audit_skb_queue;

void fn1(void) {
  audit_skb_queue = (lock); // expected-error {{use of undeclared identifier 'lock'}}
}

void fn2(void) {
  audit_skb_queue + (lock); // expected-error {{use of undeclared identifier 'lock'}}
}
