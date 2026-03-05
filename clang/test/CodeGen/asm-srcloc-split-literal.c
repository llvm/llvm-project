/// Test that inline asm source location corresponds to the actual
/// instruction line, not the first line of the asm block.
///
/// RUN: not %clang_cc1 -triple x86_64-pc-linux-gnu -emit-obj %s 2>&1 | FileCheck %s

// #include <stdint.h>
// #include <string.h>

void *memset(void *dest, int c, int n)__attribute__((naked));
void *memset(void *dest, int c, int n) {
    __asm__(
        "\t"            // <-- line with only a tab
        "xchg %eax, %eax\n" // <-- A valid instruction
        "\t"            // <-- line with only a tab
        "mov rdi, 1\n"  // <-- An invalid instruction
    );
}

int main() { return 0; }

// CHECK: error: unknown use of instruction mnemonic
// CHECK-NEXT: mov rdi, 1

