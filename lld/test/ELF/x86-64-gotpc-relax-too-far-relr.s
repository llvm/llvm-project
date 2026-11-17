# REQUIRES: x86
## Test that relative relocations added when unrelaxing GOTPCREL relocations
## during layout optimization (relaxOnce) are emitted, even if .rela.dyn or
## .relr.dyn is empty before relaxOnce.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld -pie --pack-dyn-relocs=relr --section-start=.text=0x10000 --section-start=.got=0x20000 --section-start=.data=0x100000000 %t.o -o %t
# RUN: llvm-readelf -d -r %t | FileCheck %s --check-prefix=RELR
# RUN: ld.lld -pie --section-start=.text=0x10000 --section-start=.got=0x20000 --section-start=.data=0x100000000 %t.o -o %t.rela
# RUN: llvm-readelf -d -r %t.rela | FileCheck %s --check-prefix=RELA

# RELR:      (RELRSZ) 8 (bytes)
# RELR:      Relocation section '.relr.dyn' at offset {{.*}} contains 1 entries:
# RELR-NEXT: Index: Entry Address Symbolic Address
# RELR-NEXT: 0000: 0000000000020000 0000000000020000

# RELA:      (RELASZ) 24 (bytes)
# RELA:      (NULL) 0x0
# RELA:      Relocation section '.rela.dyn' at offset {{.*}} contains 1 entries:
# RELA-NEXT: Offset Info Type Symbol's Value Symbol's Name + Addend
# RELA-NEXT: 0000000000020000 0000000000000008 R_X86_64_RELATIVE 100000000

.text
.globl _start
_start:
  movq foo@GOTPCREL(%rip), %rax

.section .data,"aw",@progbits
.globl foo
foo:
  .quad 0
