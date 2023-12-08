# REQUIRES: mips
# Check LA25 stubs creation with caller in different Output Section to callee.
# This stub code is necessary when non-PIC code calls PIC function.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:   %p/Inputs/mips-fpic.s -o %t-fpic.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:   %p/Inputs/mips-fnpic.s -o %t-fnpic.o
# RUN: ld.lld -r %t-fpic.o %t-fnpic.o -o %t-sto-pic.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:   %p/Inputs/mips-pic.s -o %t-pic.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t-npic.o
# RUN: ld.lld %t-npic.o %t-pic.o %t-sto-pic.o -o %t.exe
# RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t.exe | FileCheck %s

# CHECK: Disassembly of section .text:
# CHECK-EMPTY:
# CHECK-NEXT: <__LA25Thunk_foo1a>:
# CHECK-NEXT:    lui     $25, 0x2
# CHECK-NEXT:    j       {{.*}} <foo1a>
# CHECK-NEXT:    addiu   $25, $25, {{.*}}

# CHECK: <__LA25Thunk_foo1b>:
# CHECK-NEXT:    lui     $25, 0x2
# CHECK-NEXT:    j       {{.*}} <foo1b>
# CHECK-NEXT:    addiu   $25, $25, {{.*}}

# CHECK: <foo1a>:
# CHECK-NEXT:    nop

# CHECK: <foo1b>:
# CHECK-NEXT:    nop

# CHECK: <__LA25Thunk_foo2>:
# CHECK-NEXT:    lui     $25, 0x2
# CHECK-NEXT:    j       {{.*}} <foo2>
# CHECK-NEXT:    addiu   $25, $25, {{.*}}

# CHECK: <foo2>:
# CHECK-NEXT:    nop

# CHECK: <__LA25Thunk_fpic>:
# CHECK-NEXT:    lui     $25, 0x2
# CHECK-NEXT:    j       {{.*}} <fpic>
# CHECK-NEXT:    addiu   $25, $25, {{.*}}

# CHECK: <fpic>:
# CHECK-NEXT:    nop

# CHECK: <fnpic>:
# CHECK-NEXT:    nop
# CHECK-EMPTY:
# CHECK-NEXT: Disassembly of section .differentos:
# CHECK-EMPTY:
# CHECK-NEXT: <__start>:
# CHECK-NEXT:    jal     {{.*}} <__LA25Thunk_foo1a>
# CHECK-NEXT:    nop
# CHECK-NEXT:    jal     {{.*}} <__LA25Thunk_foo2>
# CHECK-NEXT:    nop
# CHECK-NEXT:    jal     {{.*}} <__LA25Thunk_foo1b>
# CHECK-NEXT:    nop
# CHECK-NEXT:    jal     {{.*}} <__LA25Thunk_foo2>
# CHECK-NEXT:    nop
# CHECK-NEXT:    jal     {{.*}} <__LA25Thunk_fpic>
# CHECK-NEXT:    nop
# CHECK-NEXT:    jal     {{.*}} <fnpic>

  .section .differentos, "ax", %progbits
  .globl __start
__start:
  jal foo1a
  jal foo2
  jal foo1b
  jal foo2
  jal fpic
  jal fnpic
