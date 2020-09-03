# REQUIRES: ppc

# RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %s -o %t
# RUN: ld.lld %t -o %t2
# RUN: llvm-objdump -d --no-show-raw-insn %t2 | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %s -o %t
# RUN: ld.lld %t -o %t2
# RUN: llvm-objdump -d --no-show-raw-insn %t2 | FileCheck %s

# CHECK: Disassembly of section .text:
# CHECK-EMPTY:

.text
.global _start
_start:
.Lfoo:
  li      0,1
  li      3,42
  sc

# CHECK: 10010158: li 0, 1
# CHECK: 1001015c: li 3, 42
# CHECK: 10010160: sc

.global bar
bar:
  bl _start
  nop
  bl .Lfoo
  nop
  blr

# CHECK:      10010164: bl 0x10010158
# CHECK-NEXT:           nop
# CHECK-NEXT: 1001016c: bl 0x10010158
# CHECK-NEXT:           nop
# CHECK-NEXT:           blr
