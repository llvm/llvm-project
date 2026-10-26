// RUN: %clang_cc1 -triple riscv64-unknown-openbsd -target-feature +v -ffreestanding -fsyntax-only -verify=openbsd %s
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +v -ffreestanding -fsyntax-only -verify=linux %s

// REQUIRES: riscv-registered-target

// RVV intrinsics with a 64-bit scalar/pointer operand (e.g. vse64) must use
// the target's actual uint64_t/int64_t type, not always "unsigned long":
// OpenBSD defines uint64_t as "unsigned long long" on every architecture,
// while riscv64-linux (LP64) defines it as "unsigned long".

#include <stdint.h>
#include <riscv_vector.h>

// uint64_t* must be accepted on every target, regardless of whether the
// target's uint64_t happens to be "unsigned long" or "unsigned long long".
void test_uint64_ok(uint64_t *p, vuint64m1_t v, size_t vl) {
  __riscv_vse64_v_u64m1(p, v, vl);
}

// "unsigned long *" is only the right type for the pointee on riscv64-linux
// (where uint64_t is "unsigned long"); on OpenBSD, uint64_t is
// "unsigned long long", so this should be an incompatible pointer type.
void test_unsigned_long(unsigned long *p, vuint64m1_t v, size_t vl) {
  __riscv_vse64_v_u64m1(p, v, vl);
  // openbsd-error@-1 {{incompatible pointer types passing 'unsigned long *' to parameter of type 'unsigned long long *'}}
  // openbsd-note@-2 {{passing argument to parameter here}}
}

// Conversely, "unsigned long long *" is only correct on OpenBSD; on
// riscv64-linux, uint64_t is "unsigned long", so this should fail there.
void test_unsigned_long_long(unsigned long long *p, vuint64m1_t v, size_t vl) {
  __riscv_vse64_v_u64m1(p, v, vl);
  // linux-error@-1 {{incompatible pointer types passing 'unsigned long long *' to parameter of type 'unsigned long *'}}
  // linux-note@-2 {{passing argument to parameter here}}
}
