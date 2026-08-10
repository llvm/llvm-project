// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple=i386-unknown-unknown -S -ffreestanding -o - | FileCheck %s --check-prefix=X86
// RUN: %clang_cc1 %s -triple=x86_64-unknown-unknown -S -ffreestanding -o - | FileCheck %s --check-prefix=X64

#include <immintrin.h>

// The ebx may be use for base pointer, we need to restore it in time.
void ssc_mark(void) {
// X86-LABEL: ssc_mark
// X86: #APP
// X86: movl    %ebx, %eax
// X86: movl    $9, %ebx
// X86: .byte   100
// X86: .byte   103
// X86: .byte   144
// X86: movl    %eax, %ebx
// X86: #NO_APP

// X64-LABEL: ssc_mark
// X64: #APP
// X64: movq    %rbx, %rax
// X64: movl    $9, %ebx
// X64: .byte   100
// X64: .byte   103
// X64: .byte   144
// X64: movq    %rax, %rbx
// X64: #NO_APP
  __SSC_MARK(0x9);
}
