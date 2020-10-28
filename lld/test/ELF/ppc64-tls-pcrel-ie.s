# REQUIRES: ppc

# RUN: split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=powerpc64le %t/asm -o %t.o
# RUN: llvm-mc -filetype=obj -triple=powerpc64le %t/defs -o %t-defs.o
# RUN: ld.lld --shared %t-defs.o --soname=t-defs -o %t-defs.so
# RUN: ld.lld -T %t/lds %t.o %t-defs.so -o %t-ie
# RUN: ld.lld -T %t/lds %t.o %t-defs.o -o %t-le

# RUN: llvm-readelf -r %t-ie | FileCheck %s --check-prefix=IE-RELOC
# RUN: llvm-readelf -s %t-ie | FileCheck %s --check-prefix=IE-SYM
# RUN: llvm-readelf -x .got %t-ie | FileCheck %s --check-prefix=IE-GOT
# RUN: llvm-objdump -d --no-show-raw-insn --mcpu=pwr10 %t-ie | FileCheck %s --check-prefix=IE

# RUN: llvm-readelf -r %t-le | FileCheck %s --check-prefix=LE-RELOC
# RUN: llvm-readelf -s %t-le | FileCheck %s --check-prefix=LE-SYM
# RUN: llvm-readelf -x .got %t-le 2>&1 | FileCheck %s --check-prefix=LE-GOT
# RUN: llvm-objdump -d --no-show-raw-insn --mcpu=pwr10 %t-le | FileCheck %s --check-prefix=LE

## This test checks the Initial Exec PC Relative TLS implementation.
## The IE version checks that the relocations are generated correctly.
## The LE version checks that the Initial Exec to Local Exec relaxation is
## done correctly.

#--- lds
SECTIONS {
  .text_addr 0x1001000 : { *(.text_addr) }
  .text_val 0x1002000 : { *(.text_val) }
  .text_twoval 0x1003000 : { *(.text_twoval) }
  .text_incrval 0x1004000 : { *(.text_incrval) }
}

#--- defs
.section .tbss,"awT",@nobits
.globl	x
x:
	.long	0
.globl	y
y:
	.long	0

#--- asm
# IE-RELOC: Relocation section '.rela.dyn' at offset 0x10090 contains 2 entries:
# IE-RELOC: 00000000010040d8  0000000100000049 R_PPC64_TPREL64        0000000000000000 x + 0
# IE-RELOC: 00000000010040e0  0000000200000049 R_PPC64_TPREL64        0000000000000000 y + 0

# IE-SYM:   Symbol table '.dynsym' contains 3 entries:
# IE-SYM:   1: 0000000000000000     0 TLS     GLOBAL DEFAULT   UND x
# IE-SYM:   2: 0000000000000000     0 TLS     GLOBAL DEFAULT   UND y

# IE-GOT:      Hex dump of section '.got':
# IE-GOT-NEXT: 0x010040d8 d8c00001 00000000 00000000 00000000

# LE-RELOC: There are no relocations in this file.

# LE-SYM: Symbol table '.symtab' contains 7 entries:
# LE-SYM: 5: 0000000000000000     0 TLS     GLOBAL DEFAULT     6 x
# LE-SYM: 6: 0000000000000004     0 TLS     GLOBAL DEFAULT     6 y

# LE-GOT: could not find section '.got'

# IE-LABEL: <IEAddr>:
# IE-NEXT:    pld 3, 12504(0), 1
# IE-NEXT:    add 3, 3, 13
# IE-NEXT:    blr
# LE-LABEL: <IEAddr>:
# LE-NEXT:    paddi 3, 13, -28672, 0
# LE-NEXT:    nop
# LE-NEXT:    blr
.section .text_addr, "ax", %progbits
IEAddr:
	pld 3, x@got@tprel@pcrel(0), 1
	add 3, 3, x@tls@pcrel
	blr

# IE-LABEL: <IEVal>:
# IE-NEXT:    pld 3, 8408(0), 1
# IE-NEXT:    lwzx 3, 3, 13
# IE-NEXT:    blr
# LE-LABEL: <IEVal>:
# LE-NEXT:    paddi 3, 13, -28672, 0
# LE-NEXT:    lwz 3, 0(3)
# LE-NEXT:    blr
.section .text_val, "ax", %progbits
IEVal:
	pld 3, x@got@tprel@pcrel(0), 1
	lwzx 3, 3, x@tls@pcrel
	blr

# IE-LABEL: <IETwoVal>:
# IE-NEXT:    pld 3, 4312(0), 1
# IE-NEXT:    pld 4, 4312(0), 1
# IE-NEXT:    lwzx 3, 3, 13
# IE-NEXT:    lwzx 4, 4, 13
# IE-NEXT:    blr
# LE-LABEL: <IETwoVal>:
# LE-NEXT:    paddi 3, 13, -28672, 0
# LE-NEXT:    paddi 4, 13, -28668, 0
# LE-NEXT:    lwz 3, 0(3)
# LE-NEXT:    lwz 4, 0(4)
# LE-NEXT:    blr
.section .text_twoval, "ax", %progbits
IETwoVal:
	pld 3, x@got@tprel@pcrel(0), 1
	pld 4, y@got@tprel@pcrel(0), 1
	lwzx 3, 3, x@tls@pcrel
	lwzx 4, 4, y@tls@pcrel
	blr

# IE-LABEL: <IEIncrementVal>:
# IE-NEXT:    pld 4, 224(0), 1
# IE-NEXT:    lwzx 3, 4, 13
# IE-NEXT:    stwx 3, 4, 13
# IE-NEXT:    blr
# LE-LABEL: <IEIncrementVal>:
# LE-NEXT:    paddi 4, 13, -28668, 0
# LE-NEXT:    lwz 3, 0(4)
# LE-NEXT:    stw 3, 0(4)
# LE-NEXT:    blr
.section .text_incrval, "ax", %progbits
IEIncrementVal:
	pld 4, y@got@tprel@pcrel(0), 1
	lwzx 3, 4, y@tls@pcrel
	stwx 3, 4, y@tls@pcrel
	blr
