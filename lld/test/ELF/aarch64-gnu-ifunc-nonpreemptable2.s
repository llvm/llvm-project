# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64-none-linux-gnu %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readelf -S -s %t | FileCheck %s --check-prefix=SEC
# RUN: llvm-readelf -x .rodata -x .data %t | FileCheck --check-prefix=HEX %s
# RUN: llvm-readobj -r %t | FileCheck %s --check-prefix=RELOC

## ifunc is a non-preemptable STT_GNU_IFUNC. Check we create a canonical PLT
## and redirect .rodata and .data references to it.

# SEC: .text    PROGBITS 0000000000210178
# SEC: .got.plt PROGBITS 0000000000220198
# SEC: 0000000000210180 0 FUNC GLOBAL DEFAULT 4 ifunc

## .rodata[0] and .data[0] store the address of the canonical PLT.
# HEX:      section '.rodata':
# HEX-NEXT: 0x00200170 80012100 00000000
# HEX:      section '.data':
# HEX-NEXT: 0x00220190 80012100 00000000

# RELOC:      .rela.dyn {
# RELOC-NEXT:   0x220198 R_AARCH64_IRELATIVE - 0x210178
# RELOC-NEXT: }

.globl ifunc
.type ifunc,@gnu_indirect_function
ifunc:
  ret

.rodata
.p2align 3
.xword ifunc

.data
.p2align 3
.xword ifunc
