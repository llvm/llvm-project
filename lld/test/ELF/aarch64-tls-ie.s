# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-freebsd %p/Inputs/aarch64-tls-ie.s -o %tdso.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-freebsd %s -o %tmain.o
# RUN: ld.lld -shared -soname=tdso.so %tdso.o -o %tdso.so
# RUN: ld.lld --hash-style=sysv %tmain.o %tdso.so -o %tout
# RUN: llvm-objdump -d --no-show-raw-insn %tout | FileCheck %s
# RUN: llvm-readobj -S -r %tout | FileCheck -check-prefix=RELOC %s

# RELOC:      Section {
# RELOC:        Index:
# RELOC:        Name: .got
# RELOC-NEXT:   Type: SHT_PROGBITS
# RELOC-NEXT:   Flags [
# RELOC-NEXT:     SHF_ALLOC
# RELOC-NEXT:     SHF_WRITE
# RELOC-NEXT:   ]
# RELOC-NEXT:   Address: 0x220338
# RELOC-NEXT:   Offset: 0x338
# RELOC-NEXT:   Size: 16
# RELOC-NEXT:   Link: 0
# RELOC-NEXT:   Info: 0
# RELOC-NEXT:   AddressAlignment: 8
# RELOC-NEXT:   EntrySize: 0
# RELOC-NEXT: }
# RELOC:      Relocations [
# RELOC-NEXT:  Section ({{.*}}) .rela.dyn {
# RELOC-NEXT:    0x220338 R_AARCH64_TLS_TPREL64 bar 0x0
# RELOC-NEXT:    0x220340 R_AARCH64_TLS_TPREL64 foo 0x0
# RELOC-NEXT:  }
# RELOC-NEXT:]

## Page(0x2200B0) - Page(0x210000) = 0x10000 = 65536
## 0x2200B0 & 0xff8 = 0xB0 = 176
## Page(0x2200B8) - Page(0x210000) = 0x10000 = 65536
## 0x2200B8 & 0xff8 = 0xB8 = 184
# CHECK:     <_start>:
# CHECK-NEXT: 210278: adrp x0, 0x220000
# CHECK-NEXT: 21027c: ldr  x0, [x0, #824]
# CHECK-NEXT: 210280: adrp x0, 0x220000
# CHECK-NEXT: 210284: ldr  x0, [x0, #832]

.globl _start
_start:
 adrp x0, :gottprel:bar
 ldr x0, [x0, #:gottprel_lo12:bar]

 adrp x0, :gottprel:foo
 ldr x0, [x0, #:gottprel_lo12:foo]
