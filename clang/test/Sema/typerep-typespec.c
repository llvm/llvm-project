// RUN: %clang_cc1 -std=c11 %s -fsyntax-only -verify
// REQUIRES: asserts

struct dispatch_object_s;
void _dispatch_queue_get_head(struct dispatch_object_s *volatile dq_items_head) {
  (_Atomic __typeof__(dq_items_head) *)0; // expected-warning{{expression result unused}}
}
void g(void) {
  (_Atomic __typeof__(struct dispatch_object_s *volatile) *)0; // expected-warning{{expression result unused}}
}
