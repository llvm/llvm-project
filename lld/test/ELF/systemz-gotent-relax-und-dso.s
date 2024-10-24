# REQUIRES: systemz
# RUN: llvm-mc -filetype=obj -triple=s390x-unknown-linux %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=s390x-unknown-linux %S/Inputs/gotpc-relax-und-dso.s -o %tdso.o
# RUN: ld.lld -shared %tdso.o -soname=t.so -o %t.so
# RUN: ld.lld --hash-style=sysv -shared %t.o %t.so -o %t
# RUN: llvm-readelf -r %t | FileCheck --check-prefix=RELOC %s
# RUN: llvm-objdump --no-print-imm-hex -d %t | FileCheck --check-prefix=DISASM %s

# RELOC-LABEL: Relocation section '.rela.dyn' at offset {{.*}} contains 3 entries:
# RELOC: 0000000000002428 000000010000000a R_390_GLOB_DAT 0000000000001308 foo + 0
# RELOC: 0000000000002430 000000030000000a R_390_GLOB_DAT 0000000000000000 und + 0
# RELOC: 0000000000002438 000000040000000a R_390_GLOB_DAT 0000000000000000 dsofoo + 0

# DISASM:      Disassembly of section .text:
# DISASM-EMPTY:
# DISASM-NEXT: <foo>:
# DISASM-NEXT:     nop     0
# DISASM:      <hid>:
# DISASM-NEXT:     nop     0
# DISASM:      <_start>:
# DISASM-NEXT:    lgrl    %r1, 0x2430
# DISASM-NEXT:    lgrl    %r1, 0x2430
# DISASM-NEXT:    lgrl    %r1, 0x2438
# DISASM-NEXT:    lgrl    %r1, 0x2438
# DISASM-NEXT:    larl    %r1, 0x130c
# DISASM-NEXT:    larl    %r1, 0x130c
# DISASM-NEXT:    lgrl    %r1, 0x2428
# DISASM-NEXT:    lgrl    %r1, 0x2428
# DISASM-NEXT:    lgrl    %r1, 0x2430
# DISASM-NEXT:    lgrl    %r1, 0x2430
# DISASM-NEXT:    lgrl    %r1, 0x2438
# DISASM-NEXT:    lgrl    %r1, 0x2438
# DISASM-NEXT:    larl    %r1, 0x130c
# DISASM-NEXT:    larl    %r1, 0x130c
# DISASM-NEXT:    lgrl    %r1, 0x2428
# DISASM-NEXT:    lgrl    %r1, 0x2428

.text
.globl foo
.type foo, @function
foo:
 nop

.globl hid
.hidden hid
.type hid, @function
hid:
 nop

.globl _start
.type _start, @function
_start:
 lgrl %r1, und@GOT
 lgrl %r1, und@GOT
 lgrl %r1, dsofoo@GOT
 lgrl %r1, dsofoo@GOT
 lgrl %r1, hid@GOT
 lgrl %r1, hid@GOT
 lgrl %r1, foo@GOT
 lgrl %r1, foo@GOT
 lgrl %r1, und@GOT
 lgrl %r1, und@GOT
 lgrl %r1, dsofoo@GOT
 lgrl %r1, dsofoo@GOT
 lgrl %r1, hid@GOT
 lgrl %r1, hid@GOT
 lgrl %r1, foo@GOT
 lgrl %r1, foo@GOT
