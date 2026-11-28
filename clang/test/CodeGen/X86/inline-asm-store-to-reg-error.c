// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -verify -DASM_GOTO %s -o /dev/null
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -verify %s -o /dev/null

// No integer type is 256 bits wide, so a 32-byte struct can't come back from
// an asm in registers. That's reported once, at the output that can't. Code
// generation stops after its first error, so each case is its own run.

struct S32 { long a, b, c, d; };

#ifdef ASM_GOTO

// Not once for each place an asm goto stores its outputs.
void test_asm_goto(struct S32 *p) {
  asm goto("" : "=r"(*p) : : : l1, l2); // expected-error {{impossible constraint in asm: cannot store value into a register}}
l1:
l2:;
}

#else

// Not at the output before it, which isn't returned in a register.
void test_after_memory_output(int *m, struct S32 *p) {
  asm(""
      : "=m"(*m),
        "=r"(*p)); // expected-error {{impossible constraint in asm: cannot store value into a register}}
}

#endif
