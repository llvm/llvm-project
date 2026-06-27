// RUN: %clang_cc1 -triple=x86_64-linux-gnu -verify -emit-llvm-only %s
// RUN: %clang_cc1 -triple=riscv64-linux-gnu -verify -emit-llvm-only %s

// A musttail Indirect argument is forwarded through the matching incoming
// parameter, which requires an in-memory source. A wide _BitInt has scalar
// evaluation kind, so the argument is a scalar value with no source storage
// to forward, regardless of value category. Such a call is rejected rather
// than routed through a caller-frame temp that dangles past the tail call.

typedef _BitInt(256) BI;
BI cee(BI x);
BI pee(BI a) {
  // expected-error@+1 {{'musttail' call requires passing an argument by reference, but the source does not have an addressable storage and would alias the caller's frame}}
  __attribute__((musttail)) return cee(a);
}

// An aggregate lvalue has addressable storage to forward, so it is accepted.
// Confirms the diagnostic is specific to the no-source case.
struct Big {
  unsigned long long a, b, c, d;
};
struct Big cee_ok(struct Big x);
struct Big pee_ok(struct Big a) {
  __attribute__((musttail)) return cee_ok(a);
}
