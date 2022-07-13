# REQUIRES: ppc

# RUN: llvm-mc -filetype=obj -triple=powerpc64le %s -o %t.o
# RUN: llvm-readobj -r %t.o | FileCheck --check-prefix=INPUT-REL %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.o | FileCheck --check-prefix=INPUT-ASM %s

# RUN: ld.lld --shared %t.o --soname=t.so -o %t.so
# RUN: llvm-readelf -r %t.so | FileCheck --check-prefix=SO-REL %s
# RUN: llvm-readelf -x .got %t.so | FileCheck --check-prefix=SO-GOT %s
# RUN: llvm-readelf -s %t.so | FileCheck --check-prefix=SO-SYM %s

## Test to make sure that the first element of a GOT section is the tocbase .TOC.

# INPUT-REL:          Section (3) .rela.text {
# INPUT-REL-NEXT:       0x0 R_PPC64_GOT_PCREL34 glob_int 0x0
# INPUT-REL-NEXT:       0x0 R_PPC64_PCREL_OPT - 0x8
# INPUT-REL-NEXT:     }

# INPUT-ASM-LABEL:    <test>:
# INPUT-ASM:            pld 3, 0(0), 1
# INPUT-ASM-NEXT:       lwa 3, 0(3)
# INPUT-ASM-NEXT:       blr

# SO-REL: Relocation section '.rela.dyn'
# SO-REL:   0000000000020390  0000000100000014 R_PPC64_GLOB_DAT       00000000000102d0 glob_int + 0

# SO-GOT: Hex dump of section '.got':
# SO-GOT:   0x00020388 88830200 00000000 00000000 00000000

# SO-SYM: Symbol table '.symtab' contains 4 entries:
# SO-SYM:   3: 00000000000102d0     4 NOTYPE  GLOBAL DEFAULT     6 glob_int

test:
	pld 3, glob_int@got@pcrel(0), 1
.Lpcrel0:
	.reloc .Lpcrel0-8,R_PPC64_PCREL_OPT,.-(.Lpcrel0-8)
	lwa 3, 0(3)
	blr

	.globl	glob_int
	.p2align	2
glob_int:
	.long	0
	.size	glob_int, 4
