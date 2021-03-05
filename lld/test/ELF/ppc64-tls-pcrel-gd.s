# REQUIRES: ppc
# RUN: split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=powerpc64le %t/asm -o %t.o
# RUN: llvm-mc -filetype=obj -triple=powerpc64le %t/defs -o %t-defs.o
# RUN: ld.lld --shared %t-defs.o --soname=t-defs -o %t-defs.so
# RUN: ld.lld -T %t/lds --shared %t.o -o %t-gd.so
# RUN: ld.lld -T %t/lds %t.o %t-defs.so -o %t-gdtoie
# RUN: ld.lld -T %t/lds %t.o %t-defs.o -o %t-gdtole

# RUN: llvm-readelf -r %t-gd.so | FileCheck %s --check-prefix=GD-RELOC
# RUN: llvm-readelf -s %t-gd.so | FileCheck %s --check-prefix=GD-SYM
# RUN: llvm-objdump -d --no-show-raw-insn --mcpu=pwr10 %t-gd.so | FileCheck %s --check-prefix=GD

# RUN: llvm-readelf -r %t-gdtoie | FileCheck %s --check-prefix=GDTOIE-RELOC
# RUN: llvm-readelf -s %t-gdtoie | FileCheck %s --check-prefix=GDTOIE-SYM
# RUN: llvm-objdump -d --no-show-raw-insn --mcpu=pwr10 %t-gdtoie | FileCheck %s --check-prefix=GDTOIE

# RUN: llvm-readelf -r %t-gdtole | FileCheck %s --check-prefix=GDTOLE-RELOC
# RUN: llvm-readelf -s %t-gdtole | FileCheck %s --check-prefix=GDTOLE-SYM
# RUN: llvm-objdump -d --no-show-raw-insn --mcpu=pwr10 %t-gdtole | FileCheck %s --check-prefix=GDTOLE

## This test checks the General Dynamic PC Relative TLS implementation for lld.
## GD - General Dynamic with no relaxation possible
## GDTOIE - General Dynamic relaxed to Initial Exec
## GDTOLE - General Dynamic relaxed to Local Exec

#--- lds
SECTIONS {
  .text_addr 0x1001000 : { *(.text_addr) }
}

#--- defs
.section .tbss,"awT",@nobits
.globl  x
x:
  .long 0
.globl  y
y:
  .long 0

#--- asm

# GD-RELOC: Relocation section '.rela.dyn' at offset 0x100b8 contains 4 entries:
# GD-RELOC: 0000000001001170  0000000100000044 R_PPC64_DTPMOD64       0000000000000000 x + 0
# GD-RELOC: 0000000001001178  000000010000004e R_PPC64_DTPREL64       0000000000000000 x + 0
# GD-RELOC: 0000000001001180  0000000300000044 R_PPC64_DTPMOD64       0000000000000000 y + 0
# GD-RELOC: 0000000001001188  000000030000004e R_PPC64_DTPREL64       0000000000000000 y + 0

# GD-SYM:   Symbol table '.dynsym' contains 4 entries:
# GD-SYM:   0000000000000000     0 TLS     GLOBAL DEFAULT   UND x
# GD-SYM:   0000000000000000     0 TLS     GLOBAL DEFAULT   UND y


# GDTOIE-RELOC: Relocation section '.rela.dyn' at offset 0x{{.*}} contains 2 entries:
# GDTOIE-RELOC: 00000000010010e0  0000000100000049 R_PPC64_TPREL64        0000000000000000 x + 0
# GDTOIE-RELOC: 00000000010010e8  0000000300000049 R_PPC64_TPREL64        0000000000000000 y + 0

# GDTOIE-SYM: Symbol table '.dynsym' contains 4 entries:
# GDTOIE-SYM:   0000000000000000     0 TLS     GLOBAL DEFAULT   UND x
# GDTOIE-SYM:   0000000000000000     0 TLS     GLOBAL DEFAULT   UND y


# GDTOLE-RELOC: There are no relocations in this file.

# GDTOLE-SYM: Symbol table '.symtab' contains 5 entries:
# GDTOLE-SYM: 0000000000000000     0 TLS     GLOBAL DEFAULT     3 x
# GDTOLE-SYM: 0000000000000004     0 TLS     GLOBAL DEFAULT     3 y

# GD-LABEL: <GDTwoVal>:
# GD-NEXT:    paddi 3, 0, 368, 1
# GD-NEXT:    bl
# GD-NEXT:    paddi 3, 0, 372, 1
# GD-NEXT:    bl
# GD-NEXT:    blr
# GDTOIE-LABEL: <GDTwoVal>:
# GDTOIE-NEXT:    pld 3, 224(0), 1
# GDTOIE-NEXT:    add 3, 3, 13
# GDTOIE-NEXT:    pld 3, 220(0), 1
# GDTOIE-NEXT:    add 3, 3, 13
# GDTOIE-NEXT:    blr
# GDTOLE-LABEL: <GDTwoVal>:
# GDTOLE-NEXT:    paddi 3, 13, -28672, 0
# GDTOLE-NEXT:    nop
# GDTOLE-NEXT:    paddi 3, 13, -28668, 0
# GDTOLE-NEXT:    nop
# GDTOLE-NEXT:    blr
.section .text_addr, "ax", %progbits
GDTwoVal:
  paddi 3, 0, x@got@tlsgd@pcrel, 1
  bl __tls_get_addr@notoc(x@tlsgd)
  paddi 3, 0, y@got@tlsgd@pcrel, 1
  bl __tls_get_addr@notoc(y@tlsgd)
  blr
