# REQUIRES: ppc
# RUN: llvm-mc -filetype=obj -triple=powerpc %s -o %t.o
# RUN: ld.lld -pie %t.o -o %t
# RUN: llvm-readobj -r %t | FileCheck --check-prefix=RELOC %s
# RUN: llvm-readelf -s %t | FileCheck --check-prefix=SYM %s
# RUN: llvm-readelf -x .got2 -x .got.plt %t | FileCheck --check-prefix=HEX %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

# RUN: ld.lld -pie %t.o -o %t2 --apply-dynamic-relocs
# RUN: llvm-readelf -x .got2 -x .got.plt %t2 | FileCheck --check-prefix=HEX2 %s

# RELOC:      .rela.dyn {
# RELOC-NEXT:   0x3022C R_PPC_RELATIVE - 0x101A0
# RELOC-NEXT:   0x30230 R_PPC_IRELATIVE - 0x10188
# RELOC-NEXT: }

# SYM: 000101a0 0 FUNC GLOBAL DEFAULT {{.*}} func

# HEX:      Hex dump of section '.got2':
# HEX-NEXT: 0x0003022c 00000000 ....
# HEX:      Hex dump of section '.got.plt':
# HEX-NEXT: 0x00030230 00000000 ....

# HEX2:      Hex dump of section '.got2':
# HEX2-NEXT: 0x0003022c 000101a0 ....
# HEX2:      Hex dump of section '.got.plt':
# HEX2-NEXT: 0x00030230 00000000 ....

.section .got2,"aw"
.long func

# CHECK:      Disassembly of section .text:
# CHECK:      <.text>:
# CHECK-NEXT: 10188: blr
# CHECK:      <_start>:
# CHECK-NEXT:        bl 0x10190
# CHECK-EMPTY:
# CHECK-NEXT: <00008000.got2.plt_pic32.func>:
## 0x30230 - 0x3022c - 0x8000 = -32764
# CHECK-NEXT: 10190: lwz 11, -32764(30)
# CHECK-NEXT:        mtctr 11
# CHECK-NEXT:        bctr
# CHECK-NEXT:        nop
# CHECK:      Disassembly of section .glink:
# CHECK:      <func>:
# CHECK-NEXT: 101a0: lwz 11, -32764(30)
# CHECK-NEXT:        mtctr 11
# CHECK-NEXT:        bctr
# CHECK-NEXT:        nop

.text
.globl func
.type func, @gnu_indirect_function
func:
  blr

.globl _start
_start:
  bl func+0x8000@plt
