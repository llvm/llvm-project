// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=CL2.0
// expected-no-diagnostics

__kernel void k1(queue_t q1, queue_t q2) {
  queue_t q[] = {q1, q2};
}

__kernel void k2(read_only pipe int p) {
  reserve_id_t i1 = reserve_read_pipe(p, 1);
  reserve_id_t i2 = reserve_read_pipe(p, 1);
  reserve_id_t i[] = {i1, i2};
}

event_t create_event();
__kernel void k3() {
  event_t e1 = create_event();
  event_t e2 = create_event();
  event_t e[] = {e1, e2};
}

