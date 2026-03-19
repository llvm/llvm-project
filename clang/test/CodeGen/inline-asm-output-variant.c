// REQUIRES: x86-registered-target
/// AT&T input
// RUN: %clang_cc1 -triple x86_64 -S --output-asm-variant=0 %s -o - | FileCheck --check-prefix=ATT %s
// RUN: %clang_cc1 -triple x86_64 -S --output-asm-variant=1 %s -o - | FileCheck --check-prefix=INTEL %s

/// Intel input
// RUN: %clang_cc1 -triple x86_64 -S -D INTEL -mllvm -x86-asm-syntax=intel -inline-asm=intel %s -o - | FileCheck --check-prefix=INTEL %s
// RUN: %clang_cc1 -triple x86_64 -S -D INTEL -mllvm -x86-asm-syntax=intel -inline-asm=intel --output-asm-variant=1 %s -o - | FileCheck --check-prefix=INTEL %s

// ATT: movl $1, %eax
// ATT: movl $2, %eax

// INTEL: mov eax, 1
// INTEL: mov eax, 2

#ifdef INTEL
asm("mov eax, 1");
void foo() {
  asm("mov eax, 2");
}
#else
asm("mov $1, %eax");
void foo() {
  asm("mov $2, %eax");
}
#endif
