# REQUIRES: x86

# RUN: mkdir -p %t
#
# RUN: llvm-mc -filetype obj -triple x86_64-apple-darwin %s -o %t/test.o
# RUN: lld -flavor darwinnew -arch x86_64 -o %t/test -Z -L%S/Inputs/MacOSX.sdk/usr/lib -lSystem %t/test.o
#
# RUN: llvm-objdump --bind --no-show-raw-insn -d -r %t/test | FileCheck %s

# CHECK: Disassembly of section __TEXT,__text:
# CHECK: movq {{.*}} # [[ADDR:[0-9a-f]+]]

# CHECK: Bind table:
# CHECK: __DATA_CONST __got 0x[[ADDR]] pointer 0 libSystem ___nan

.section __TEXT,__text
.global _main

_main:
  movq ___nan@GOTPCREL(%rip), %rax
  ret
