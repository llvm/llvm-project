// RUN: %clang_cc1 -triple arm64-apple-darwin -fsyntax-only -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

typedef unsigned char uint8_t;

uint8_t constraint_r(uint8_t *addr) {
  uint8_t byte;

  __asm__ volatile("ldrb %0, [%1]" : "=r" (byte) : "r" (addr) : "memory");
// CHECK: warning: value size does not match register size specified by the constraint and modifier
// CHECK: note: use constraint modifier "w"
// CHECK: fix-it:{{.*}}:{8:26-8:28}:"%w0"

  return byte;
}

uint8_t constraint_r_symbolic(uint8_t *addr) {
  uint8_t byte;

  __asm__ volatile("ldrb %[s0], [%[s1]]" : [s0] "=r" (byte) : [s1] "r" (addr) : "memory");
// CHECK: warning: value size does not match register size specified by the constraint and modifier
// CHECK: note: use constraint modifier "w"
// CHECK: fix-it:{{.*}}:{19:26-19:31}:"%w[s0]"

  return byte;
}

#define PERCENT "%"

uint8_t constraint_r_symbolic_macro(uint8_t *addr) {
  uint8_t byte;

  __asm__ volatile("ldrb "PERCENT"[s0], [%[s1]]" : [s0] "=r" (byte) : [s1] "r" (addr) : "memory");
// CHECK: warning: value size does not match register size specified by the constraint and modifier
// CHECK: note: use constraint modifier "w"
// CHECK-NOT: fix-it

  return byte;
}

// CHECK: warning: value size does not match register size specified by the constraint and modifier
// CHECK: asm ("%w0 %w1 %2" : "+r" (one) : "r" (wide_two));
// CHECK: note: use constraint modifier "w"
// CHECK: fix-it:{{.*}}:{47:17-47:19}:"%w2"

void read_write_modifier0(int one, int two) {
  long wide_two = two;
  asm ("%w0 %w1 %2" : "+r" (one) : "r" (wide_two));
}

// CHECK-NOT: warning: 
void read_write_modifier1(int one, int two) {
  long wide_two = two;
  asm ("%w0 %1" : "+r" (one), "+r" (wide_two));
}
