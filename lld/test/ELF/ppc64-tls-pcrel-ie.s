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
# RUN: llvm-objdump -d --no-show-raw-insn %t-ie | FileCheck %s --check-prefix=IE

# RUN: llvm-readelf -r %t-le | FileCheck %s --check-prefix=LE-RELOC
# RUN: llvm-readelf -s %t-le | FileCheck %s --check-prefix=LE-SYM
# RUN: llvm-readelf -x .got %t-le 2>&1 | FileCheck %s --check-prefix=LE-GOT
# RUN: llvm-objdump -d --no-show-raw-insn %t-le | FileCheck %s --check-prefix=LE

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
  .text_incrval_half 0x1005000 : { *(.text_incrval_half) }
  .text_incrval_word 0x1006000 : { *(.text_incrval_word) }
  .text_incrval_float 0x1007000 : { *(.text_incrval_float) }
  .text_incrval_double 0x1008000 : { *(.text_incrval_double) }
  .text_incrval_dword 0x1009000 : { *(.text_incrval_dword) }
  .text_incrval_half_zero 0x1010000 : { *(.text_incrval_half_zero) }
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
# IE-RELOC: 00000000010100f0  0000000100000049 R_PPC64_TPREL64        0000000000000000 x + 0
# IE-RELOC-NEXT: 00000000010100f8  0000000200000049 R_PPC64_TPREL64        0000000000000000 y + 0

# IE-SYM:   Symbol table '.dynsym' contains 3 entries:
# IE-SYM:   1: 0000000000000000     0 TLS     GLOBAL DEFAULT   UND x
# IE-SYM:   2: 0000000000000000     0 TLS     GLOBAL DEFAULT   UND y

# IE-GOT:      Hex dump of section '.got':
# IE-GOT-NEXT: 0x010100e8 e8800101 00000000 00000000 00000000

# LE-RELOC: There are no relocations in this file.

# LE-SYM: Symbol table '.symtab' contains 14 entries:
# LE-SYM: 0000000000000000     0 TLS     GLOBAL DEFAULT     [[#]] x
# LE-SYM: 0000000000000004     0 TLS     GLOBAL DEFAULT     [[#]] y

# LE-GOT: could not find section '.got'

# IE-LABEL: <IEAddr>:
# IE-NEXT:    pld 3, 61680(0), 1
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

# IE-LABEL: <IEAddrCopy>:
# IE-NEXT:    pld 3, 61664(0), 1
# IE-NEXT:    add 4, 3, 13
# IE-NEXT:    blr
# LE-LABEL: <IEAddrCopy>:
# LE-NEXT:    paddi 3, 13, -28672, 0
# LE-NEXT:    mr 4, 3
# LE-NEXT:    blr
.section .text_addr, "ax", %progbits
IEAddrCopy:
	pld 3, x@got@tprel@pcrel(0), 1
	add 4, 3, x@tls@pcrel
	blr

# IE-LABEL: <IEVal>:
# IE-NEXT:    pld 3, 57584(0), 1
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
# IE-NEXT:    pld 3, 53488(0), 1
# IE-NEXT:    pld 4, 53488(0), 1
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
# IE-NEXT:    pld 4, 49400(0), 1
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

# IE-LABEL: <IEIncrementValHalf>:
# IE-NEXT:    pld 4, 45304(0), 1
# IE-NEXT:    lhax 3, 4, 13
# IE-NEXT:    sthx 3, 4, 13
# IE-NEXT:    blr
# LE-LABEL: <IEIncrementValHalf>:
# LE-NEXT:    paddi 4, 13, -28668, 0
# LE-NEXT:    lha 3, 0(4)
# LE-NEXT:    sth 3, 0(4)
# LE-NEXT:    blr
.section .text_incrval_half, "ax", %progbits
IEIncrementValHalf:
	pld 4, y@got@tprel@pcrel(0), 1
	lhax 3, 4, y@tls@pcrel
	sthx 3, 4, y@tls@pcrel
	blr

# IE-LABEL: <IEIncrementValWord>:
# IE-NEXT:    pld 4, 41208(0), 1
# IE-NEXT:    lwax 3, 4, 13
# IE-NEXT:    stwx 3, 4, 13
# IE-NEXT:    blr
# LE-LABEL: <IEIncrementValWord>:
# LE-NEXT:    paddi 4, 13, -28668, 0
# LE-NEXT:    lwa 3, 0(4)
# LE-NEXT:    stw 3, 0(4)
# LE-NEXT:    blr
.section .text_incrval_word, "ax", %progbits
IEIncrementValWord:
	pld 4, y@got@tprel@pcrel(0), 1
	lwax 3, 4, y@tls@pcrel
	stwx 3, 4, y@tls@pcrel
	blr

# IE-LABEL: <IEIncrementValFloat>:
# IE-NEXT:    pld 4, 37112(0), 1
# IE-NEXT:    lfsx 3, 4, 13
# IE-NEXT:    stfsx 3, 4, 13
# IE-NEXT:    blr
# LE-LABEL: <IEIncrementValFloat>:
# LE-NEXT:    paddi 4, 13, -28668, 0
# LE-NEXT:    lfs 3, 0(4)
# LE-NEXT:    stfs 3, 0(4)
# LE-NEXT:    blr
.section .text_incrval_float, "ax", %progbits
IEIncrementValFloat:
	pld 4, y@got@tprel@pcrel(0), 1
	lfsx 3, 4, y@tls@pcrel
	stfsx 3, 4, y@tls@pcrel
	blr

# IE-LABEL: <IEIncrementValDouble>:
# IE-NEXT:    pld 4, 33016(0), 1
# IE-NEXT:    lfdx 3, 4, 13
# IE-NEXT:    stfdx 3, 4, 13
# IE-NEXT:    blr
# LE-LABEL: <IEIncrementValDouble>:
# LE-NEXT:    paddi 4, 13, -28668, 0
# LE-NEXT:    lfd 3, 0(4)
# LE-NEXT:    stfd 3, 0(4)
# LE-NEXT:    blr
.section .text_incrval_double, "ax", %progbits
IEIncrementValDouble:
	pld 4, y@got@tprel@pcrel(0), 1
	lfdx 3, 4, y@tls@pcrel
	stfdx 3, 4, y@tls@pcrel
	blr

# IE-LABEL: <IEIncrementValDword>:
# IE-NEXT:    pld 4, 28920(0), 1
# IE-NEXT:    ldx 3, 4, 13
# IE-NEXT:    stdx 3, 4, 13
# IE-NEXT:    blr
# LE-LABEL: <IEIncrementValDword>:
# LE-NEXT:    paddi 4, 13, -28668, 0
# LE-NEXT:    ld 3, 0(4)
# LE-NEXT:    std 3, 0(4)
# LE-NEXT:    blr
.section .text_incrval_dword, "ax", %progbits
IEIncrementValDword:
	pld 4, y@got@tprel@pcrel(0), 1
	ldx 3, 4, y@tls@pcrel
	stdx 3, 4, y@tls@pcrel
	blr

# IE-LABEL: <IEIncrementValHalfZero>:
# IE-NEXT:    pld 4, 248(0), 1
# IE-NEXT:    lhzx 3, 4, 13
# IE-NEXT:    sthx 3, 4, 13
# IE-NEXT:    blr
# LE-LABEL: <IEIncrementValHalfZero>:
# LE-NEXT:    paddi 4, 13, -28668, 0
# LE-NEXT:    lhz 3, 0(4)
# LE-NEXT:    sth 3, 0(4)
# LE-NEXT:    blr
.section .text_incrval_half_zero, "ax", %progbits
IEIncrementValHalfZero:
	pld 4, y@got@tprel@pcrel(0), 1
	lhzx 3, 4, y@tls@pcrel
	sthx 3, 4, y@tls@pcrel
	blr
