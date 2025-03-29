# The second and third ADR instructions are non-local to functions
# and must be replaced with ADRP + ADD by BOLT
# Also since main and test are non-simple, we can't change it's length so we
# have to replace NOP with adrp, and if there is no nop before adr in non-simple
# function, we can't guarantee we didn't break possible jump tables, so we
# fail in non-strict mode

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --adr-relaxation=true --strict
# RUN: llvm-objdump --no-print-imm-hex -d --disassemble-symbols=main %t.bolt | FileCheck %s
# RUN: %t.bolt
# RUN: not llvm-bolt %t.exe -o %t.bolt --adr-relaxation=true \
# RUN: 2>&1 | FileCheck %s --check-prefix CHECK-ERROR

  .text
  .align 4
  .global test
  .type test, %function
test:
  adr x2, Gvar
  mov x0, xzr
  ret
  .size test, .-test

  .align 4
  .global main
  .type main, %function
main:
  adr x0, .CI
  nop
  adr x1, test
  adr x2, Gvar2
  adr x3, br
br:
  br x1
  .size main, .-main
.CI:
  .word 0xff

  .data
  .align 8
  .global Gvar
Gvar: .xword 0x0
  .global Gvar2
Gvar2: .xword 0x42
  .balign 4
jmptable:
  .word 0
  .word test - jmptable

# CHECK: <main>:
# CHECK-NEXT: adr x0, 0x{{[1-8a-f][0-9a-f]*}}
# CHECK-NEXT: adrp x1, 0x{{[1-8a-f][0-9a-f]*}}
# CHECK-NEXT: add x1, x1, #{{[1-8a-f][0-9a-f]*}}
# CHECK-NEXT: adrp x2, 0x{{[1-8a-f][0-9a-f]*}}
# CHECK-NEXT: add x2, x2, #{{[1-8a-f][0-9a-f]*}}
# CHECK-NEXT: adr x3, 0x{{[1-8a-f][0-9a-f]*}}
# CHECK-ERROR: BOLT-ERROR: Cannot relax adr in non-simple function
