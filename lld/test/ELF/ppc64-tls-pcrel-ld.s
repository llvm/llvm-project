# REQUIRES: ppc
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=powerpc64le %t/asm -o %t.o
# RUN: ld.lld -T %t/lds --shared -soname=t-ld %t.o -o %t-ld.so
# RUN: ld.lld -T %t/lds %t.o -o %t-ldtole

# RUN: llvm-readelf -r %t-ld.so | FileCheck %s --check-prefix=LD-RELOC
# RUN: llvm-readelf -s %t-ld.so | FileCheck %s --check-prefix=LD-SYM
# RUN: llvm-readelf -x .got %t-ld.so | FileCheck %s --check-prefix=LD-GOT
# RUN: llvm-objdump -d --no-show-raw-insn --mcpu=pwr10 %t-ld.so | FileCheck %s --check-prefix=LD

# RUN: llvm-readelf -r %t-ldtole | FileCheck %s --check-prefix=LDTOLE-RELOC
# RUN: llvm-readelf -s %t-ldtole | FileCheck %s --check-prefix=LDTOLE-SYM
# RUN: llvm-readelf -x .got %t-ldtole 2>&1 | FileCheck %s --check-prefix=LDTOLE-GOT
# RUN: llvm-objdump -d --no-show-raw-insn --mcpu=pwr10 %t-ldtole | FileCheck %s --check-prefix=LDTOLE

## This test checks the Local Dynamic PC Relative TLS implementation for lld.
## LD - Local Dynamic with no relaxation possible
## LDTOLE - Local Dynamic relaxed to Local Exec

# LD-RELOC: Relocation section '.rela.dyn' at offset 0x10080 contains 1 entries:
# LD-RELOC: 0000000001004168  0000000000000044 R_PPC64_DTPMOD64                  0

# LD-SYM:      Symbol table '.symtab' contains 11 entries:
# LD-SYM:      5: 0000000000000000     0 TLS     LOCAL  DEFAULT    13 x
# LD-SYM-NEXT: 6: 0000000000000004     0 TLS     LOCAL  DEFAULT    13 y

# LD-GOT:      section '.got':
# LD-GOT-NEXT: 0x01004160 60c10001 00000000 00000000 00000000
# LD-GOT-NEXT: 0x01004170 00000000 00000000

# LDTOLE-RELOC: There are no relocations in this file.

# LDTOLE-SYM:      Symbol table '.symtab' contains 9 entries:
# LDTOLE-SYM:      5: 0000000000000000     0 TLS     LOCAL  DEFAULT     6 x
# LDTOLE-SYM-NEXT: 6: 0000000000000004     0 TLS     LOCAL  DEFAULT     6 y

# LDTOLE-GOT:      section '.got':
# LDTOLE-GOT-NEXT: 0x01004020 20c00001 00000000

//--- lds
SECTIONS {
  .text_addr 0x1001000 : { *(.text_addr) }
  .text_val 0x1002000 : { *(.text_val) }
  .text_twoval 0x1003000 : { *(.text_twoval) }
  .text_incrval 0x1004000 : { *(.text_incrval) }
}

//--- asm
# LD-LABEL: <LDAddr>:
# LD:         paddi 3, 0, 12644, 1
# LD-NEXT:    bl 0x1001020
# LD-NEXT:    paddi 3, 3, -32768, 0
# LD-NEXT:    blr
# LDTOLE-LABEL: <LDAddr>:
# LDTOLE:         paddi 3, 13, 4096, 0
# LDTOLE-NEXT:    nop
# LDTOLE-NEXT:    paddi 3, 3, -32768, 0
# LDTOLE-NEXT:    blr
.section .text_addr, "ax", %progbits
LDAddr:
  ## TODO: Adding a reference to .TOC. since LLD doesn't gracefully handle the
  ## case where we define a .got section but have no references to the toc base
  ## yet.
  addis 2, 12, .TOC.-LDAddr@ha
  paddi 3, 0, x@got@tlsld@pcrel, 1
  bl __tls_get_addr@notoc(x@tlsld)
  paddi 3, 3, x@dtprel, 0
  blr

# LD-LABEL: <LDVal>:
# LD:         paddi 3, 0, 8552, 1
# LD-NEXT:    bl 0x1001020
# LD-NEXT:    paddi 3, 3, -32768, 0
# LD-NEXT:    lwz 3, 0(3)
# LD-NEXT:    blr
# LDTOLE-LABEL: <LDVal>:
# LDTOLE:         paddi 3, 13, 4096, 0
# LDTOLE-NEXT:    nop
# LDTOLE-NEXT:    paddi 3, 3, -32768, 0
# LDTOLE-NEXT:    lwz 3, 0(3)
# LDTOLE-NEXT:    blr
.section .text_val, "ax", %progbits
LDVal:
  paddi 3, 0, x@got@tlsld@pcrel, 1
  bl __tls_get_addr@notoc(x@tlsld)
  paddi 3, 3, x@dtprel, 0
  lwz 3, 0(3)
  blr

# LD-LABEL: <LDTwoVal>:
# LD:         paddi 3, 0, 4456, 1
# LD-NEXT:    bl 0x1001020
# LD-NEXT:    paddi 3, 3, -32768, 0
# LD-NEXT:    lwz 2, 0(3)
# LD-NEXT:    paddi 3, 3, -32764, 0
# LD-NEXT:    lwz 3, 0(3)
# LD-NEXT:    blr
# LDTOLE-LABEL: <LDTwoVal>:
# LDTOLE:         paddi 3, 13, 4096, 0
# LDTOLE-NEXT:    nop
# LDTOLE-NEXT:    paddi 3, 3, -32768, 0
# LDTOLE-NEXT:    lwz 2, 0(3)
# LDTOLE-NEXT:    paddi 3, 3, -32764, 0
# LDTOLE-NEXT:    lwz 3, 0(3)
# LDTOLE-NEXT:    blr
.section .text_twoval, "ax", %progbits
LDTwoVal:
  paddi 3, 0, x@got@tlsld@pcrel, 1
  bl __tls_get_addr@notoc(x@tlsld)
  paddi 3, 3, x@dtprel, 0
  lwz 2, 0(3)
  paddi 3, 3, y@dtprel, 0
  lwz 3, 0(3)
  blr

# LD-LABEL: <LDIncrementVal>:
# LD:         paddi 3, 0, 360, 1
# LD-NEXT:    bl 0x1001020
# LD-NEXT:    paddi 9, 3, -32764, 0
# LD-NEXT:    lwz 4, 0(9)
# LD-NEXT:    stw 5, 0(9)
# LD-NEXT:    blr
# LDTOLE-LABEL: <LDIncrementVal>:
# LDTOLE:         paddi 3, 13, 4096, 0
# LDTOLE-NEXT:    nop
# LDTOLE-NEXT:    paddi 9, 3, -32764, 0
# LDTOLE-NEXT:    lwz 4, 0(9)
# LDTOLE-NEXT:    stw 5, 0(9)
# LDTOLE-NEXT:    blr
.section .text_incrval, "ax", %progbits
LDIncrementVal:
  paddi 3, 0, y@got@tlsld@pcrel, 1
  bl __tls_get_addr@notoc(y@tlsld)
  paddi 9, 3, y@dtprel, 0
  lwz 4, 0(9)
  stw 5, 0(9)
  blr

.section .tbss,"awT",@nobits
x:
  .long   0
y:
  .long   0
