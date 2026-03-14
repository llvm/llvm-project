// RUN: %clang_cc1 -triple aarch64 -fsyntax-only -verify -DVERIFY %s
// RUN: %clang_cc1 -triple arm64-apple-darwin -fsyntax-only -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

typedef unsigned char uint8_t;

#ifdef VERIFY
void test_s(int i) {
  asm("" :: "s"(i)); // expected-error{{invalid input constraint 's' in asm}}

  /// Codegen error
  asm("" :: "S"(i));
  asm("" :: "S"(test_s(i))); // expected-error{{invalid type 'void' in asm input for constraint 'S'}}
}
#else
uint8_t constraint_r(uint8_t *addr) {
  uint8_t byte;

  __asm__ volatile("ldrb %0, [%1]" : "=r" (byte) : "r" (addr) : "memory");
// CHECK: warning: value size does not match register size specified by the constraint and modifier
// CHECK: note: use constraint modifier "w"
// CHECK: fix-it:{{.*}}:{[[#@LINE-3]]:26-[[#@LINE-3]]:28}:"%w0"

  return byte;
}

uint8_t constraint_r_symbolic(uint8_t *addr) {
  uint8_t byte;

  __asm__ volatile("ldrb %[s0], [%[s1]]" : [s0] "=r" (byte) : [s1] "r" (addr) : "memory");
// CHECK: warning: value size does not match register size specified by the constraint and modifier
// CHECK: note: use constraint modifier "w"
// CHECK: fix-it:{{.*}}:{[[#@LINE-3]]:26-[[#@LINE-3]]:31}:"%w[s0]"

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

void read_write_modifier0(int one, int two) {
  long wide_two = two;
  asm ("%w0 %w1 %2" : "+r" (one) : "r" (wide_two));
// CHECK: fix-it:{{.*}}:{[[#@LINE-1]]:17-[[#@LINE-1]]:19}:"%w2"
}

// CHECK-NOT: warning: 
void read_write_modifier1(int one, int two) {
  long wide_two = two;
  asm ("%w0 %1" : "+r" (one), "+r" (wide_two));
}
#endif
