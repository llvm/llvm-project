// REQUIRES: x86-registered-target
// RUN: %clang -cc1as -triple x86_64 %s -o - | FileCheck %s --check-prefix=ATT
// RUN: %clang -cc1as -triple x86_64 %s --output-asm-variant=1 -o - | FileCheck %s --check-prefix=INTEL

// ATT: movl $1, %eax
// INTEL: mov eax, 1

mov $1, %eax
