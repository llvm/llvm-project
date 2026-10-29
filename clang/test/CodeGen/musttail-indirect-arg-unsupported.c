// RUN: %clang_cc1 -triple=x86_64-linux-gnu -verify -emit-llvm-only %s
// RUN: %clang_cc1 -triple=riscv64-linux-gnu -verify -emit-llvm-only %s

// A wide _BitInt has no addressable source for indirect forwarding.
// Storing it into the incoming slot could support this case.

typedef _BitInt(256) BI;
BI cee(BI x);
BI pee(BI a) {
  // expected-error@+1 {{'musttail' call cannot safely forward this indirect argument}}
  __attribute__((musttail)) return cee(a);
}

// An aggregate lvalue has addressable storage to forward.
struct Big {
  unsigned long long a, b, c, d;
};
struct Big cee_ok(struct Big x);
struct Big pee_ok(struct Big a) {
  __attribute__((musttail)) return cee_ok(a);
}
