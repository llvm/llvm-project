# REQUIRES: ppc
## Test .got2 placed in a different output section.

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=powerpc %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=powerpc %t/b.s -o %t/b.o
# RUN: ld.lld -shared -T %t/t %t/a.o %t/b.o -o %t/a.so
# RUN: llvm-readobj -r %t/a.so | FileCheck --check-prefix=RELOC %s
# RUN: llvm-readelf -S %t/a.so | FileCheck --check-prefix=SEC %s

# RELOC:      .rela.plt {
# RELOC-NEXT:   0x1A4 R_PPC_JMP_SLOT f 0x0
# RELOC-NEXT: }

# SEC:      .got    PROGBITS 0000018c
# SEC-NEXT: .rodata PROGBITS 00000198

## .got2+0x8000-0xb0 = .rodata+4+0x8000-0xb0 = 0x198+4+0x8000-0xb0 = 65536*1-32532
# CHECK:      <_start>:
# CHECK-NEXT:          bcl 20, 31, 0x
# CHECK-NEXT:      b0: mflr 30
# CHECK-NEXT:          addis 30, 30, 1
# CHECK-NEXT:          addi 30, 30, -32532
# CHECK-NEXT:          bl {{.*}} <00008000.got2.plt_pic32.f>

## &.got[2] - (.got2+0x8000) = &.got[2] - (.rodata+4+0x8000) = 0x1A4 - (0x198+4+0x8000) = -32760
# CHECK:      <00008000.got2.plt_pic32.f>:
# CHECK-NEXT:   lwz 11, -32760(30)
# CHECK-NEXT:   mtctr 11
# CHECK-NEXT:   bctr
# CHECK-NEXT:   nop

#--- a.s
.section .rodata.cst4,"aM",@progbits,4
.long 1

.section .got2,"aw"
.long f

.text
.globl _start, f, g
_start:
  bcl 20,31,.L
.L:
  mflr 30
  addis 30, 30, .got2+0x8000-.L@ha
  addi 30, 30, .got2+0x8000-.L@l
  bl f+0x8000@plt

#--- b.s
.section .got2,"aw"
.globl f
f:
  bl f+0x8000@plt

#--- t
SECTIONS {
  .rodata : { *(.rodata .rodata.*) *(.got2) }
}
