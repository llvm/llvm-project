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
# RUN: llvm-objdump -d --no-show-raw-insn %t.exe | FileCheck %s

# CHECK: Disassembly of section .text:
# CHECK-EMPTY:
# CHECK-NEXT: __LA25Thunk_foo1a:
# CHECK-NEXT:    20000:       lui     $25, 2
# CHECK-NEXT:    20004:       j       131104 <foo1a>
# CHECK-NEXT:    20008:       addiu   $25, $25, 32
# CHECK-NEXT:    2000c:       nop

# CHECK: __LA25Thunk_foo1b:
# CHECK-NEXT:    20010:       lui     $25, 2
# CHECK-NEXT:    20014:       j       131108 <foo1b>
# CHECK-NEXT:    20018:       addiu   $25, $25, 36
# CHECK-NEXT:    2001c:       nop

# CHECK: foo1a:
# CHECK-NEXT:    20020:       nop

# CHECK: foo1b:
# CHECK-NEXT:    20024:       nop

# CHECK: __LA25Thunk_foo2:
# CHECK-NEXT:    20028:       lui     $25, 2
# CHECK-NEXT:    2002c:       j       131136 <foo2>
# CHECK-NEXT:    20030:       addiu   $25, $25, 64
# CHECK-NEXT:    20034:       nop

# CHECK: foo2:
# CHECK-NEXT:    20040:       nop

# CHECK: __LA25Thunk_fpic:
# CHECK-NEXT:    20044:       lui     $25, 2
# CHECK-NEXT:    20048:       j       131168 <fpic>
# CHECK-NEXT:    2004c:       addiu   $25, $25, 96
# CHECK-NEXT:    20050:       nop

# CHECK: fpic:
# CHECK-NEXT:    20060:       nop

# CHECK: fnpic:
# CHECK-NEXT:    20070:       nop
# CHECK-EMPTY:
# CHECK-NEXT: Disassembly of section differentos:
# CHECK-EMPTY:
# CHECK-NEXT: __start:
# CHECK-NEXT:    20074:       jal     131072 <__LA25Thunk_foo1a>
# CHECK-NEXT:    20078:       nop
# CHECK-NEXT:    2007c:       jal     131112 <__LA25Thunk_foo2>
# CHECK-NEXT:    20080:       nop
# CHECK-NEXT:    20084:       jal     131088 <__LA25Thunk_foo1b>
# CHECK-NEXT:    20088:       nop
# CHECK-NEXT:    2008c:       jal     131112 <__LA25Thunk_foo2>
# CHECK-NEXT:    20090:       nop
# CHECK-NEXT:    20094:       jal     131140 <__LA25Thunk_fpic>
# CHECK-NEXT:    20098:       nop
# CHECK-NEXT:    2009c:       jal     131184 <fnpic>
# CHECK-NEXT:    200a0:       nop

# Make sure the thunks are created properly no matter how
# objects are laid out.
#
# RUN: ld.lld %t-pic.o %t-npic.o %t-sto-pic.o -o %t.exe
# RUN: llvm-objdump -d --no-show-raw-insn %t.exe | FileCheck -check-prefix=REVERSE %s

# REVERSE: Disassembly of section .text:
# REVERSE-EMPTY:
# REVERSE-NEXT: __LA25Thunk_foo1a:
# REVERSE-NEXT:    20000:       lui     $25, 2
# REVERSE-NEXT:    20004:       j       131104 <foo1a>
# REVERSE-NEXT:    20008:       addiu   $25, $25, 32
# REVERSE-NEXT:    2000c:       nop

# REVERSE: __LA25Thunk_foo1b:
# REVERSE-NEXT:    20010:       lui     $25, 2
# REVERSE-NEXT:    20014:       j       131108 <foo1b>
# REVERSE-NEXT:    20018:       addiu   $25, $25, 36
# REVERSE-NEXT:    2001c:       nop

# REVERSE: foo1a:
# REVERSE-NEXT:    20020:       nop

# REVERSE: foo1b:
# REVERSE-NEXT:    20024:       nop

# REVERSE: __LA25Thunk_foo2:
# REVERSE-NEXT:    20028:       lui     $25, 2
# REVERSE-NEXT:    2002c:       j       131136 <foo2>
# REVERSE-NEXT:    20030:       addiu   $25, $25, 64
# REVERSE-NEXT:    20034:       nop

# REVERSE: foo2:
# REVERSE-NEXT:    20040:       nop

# REVERSE: __LA25Thunk_fpic:
# REVERSE-NEXT:    20050:       lui     $25, 2
# REVERSE-NEXT:    20054:       j       131168 <fpic>
# REVERSE-NEXT:    20058:       addiu   $25, $25, 96
# REVERSE-NEXT:    2005c:       nop

# REVERSE: fpic:
# REVERSE-NEXT:    20060:       nop

# REVERSE: fnpic:
# REVERSE-NEXT:    20070:       nop

# REVERSE: Disassembly of section differentos:
# REVERSE-EMPTY:
# REVERSE-NEXT: __start:
# REVERSE-NEXT:    20074:       jal     131072 <__LA25Thunk_foo1a>
# REVERSE-NEXT:    20078:       nop
# REVERSE-NEXT:    2007c:       jal     131112 <__LA25Thunk_foo2>
# REVERSE-NEXT:    20080:       nop
# REVERSE-NEXT:    20084:       jal     131088 <__LA25Thunk_foo1b>
# REVERSE-NEXT:    20088:       nop
# REVERSE-NEXT:    2008c:       jal     131112 <__LA25Thunk_foo2>
# REVERSE-NEXT:    20090:       nop
# REVERSE-NEXT:    20094:       jal     131152 <__LA25Thunk_fpic>
# REVERSE-NEXT:    20098:       nop
# REVERSE-NEXT:    2009c:       jal     131184 <fnpic>
# REVERSE-NEXT:    200a0:       nop

  .section differentos, "ax", %progbits
  .globl __start
__start:
  jal foo1a
  jal foo2
  jal foo1b
  jal foo2
  jal fpic
  jal fnpic
