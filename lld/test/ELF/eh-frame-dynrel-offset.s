# REQUIRES: x86
## RelocScan::scanEhSection maps .eh_frame relocation offsets to the merged
## output section. Check that the dynamic relocations it creates are not mapped
## a second time. Two files are needed: the first record lands at output offset
## 0, where a second mapping is a no-op.

# RUN: rm -rf %t && split-file %s %t && cd %t
## lld only emits dynamic relocations into a writable .eh_frame.
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-objcopy --set-section-flags .eh_frame=alloc,data a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o
# RUN: llvm-objcopy --set-section-flags .eh_frame=alloc,data b.o

# RUN: ld.lld -shared --apply-dynamic-relocs --check-dynamic-relocations a.o b.o -o rela
# RUN: llvm-readelf -S -r rela | FileCheck %s --implicit-check-not=R_X86_64

# CHECK:      .eh_frame         PROGBITS        {{0*}}[[#%x,EH:]]
# CHECK:      {{0*}}[[#%x,EH+0x12]] {{.*}} R_X86_64_RELATIVE
# CHECK-NEXT: {{0*}}[[#%x,EH+0x7a]] {{.*}} R_X86_64_RELATIVE
# CHECK-NEXT: {{0*}}[[#%x,EH+0x46]] {{.*}} R_X86_64_64 {{.*}} ext1 + 10
# CHECK-NEXT: {{0*}}[[#%x,EH+0xae]] {{.*}} R_X86_64_64 {{.*}} ext2 + 20

## .relr.dyn stores the offsets separately from .rela.dyn.
# RUN: ld.lld -shared -z pack-relative-relocs a.o b.o -o relr
# RUN: llvm-readelf -S -r relr | FileCheck %s --check-prefix=RELR --implicit-check-not=R_X86_64

# RELR:      .eh_frame         PROGBITS        {{0*}}[[#%x,EH:]]
# RELR:      {{0*}}[[#%x,EH+0x46]] {{.*}} R_X86_64_64 {{.*}} ext1 + 10
# RELR-NEXT: {{0*}}[[#%x,EH+0xae]] {{.*}} R_X86_64_64 {{.*}} ext2 + 20
# RELR:      0000: {{.*}} {{0*}}[[#%x,EH+0x12]]
# RELR-NEXT: 0001: {{.*}} {{0*}}[[#%x,EH+0x7a]]

#--- a.s
.set pers1, ext1+0x10
.text
.globl f1
f1:
  .cfi_startproc
  .cfi_personality 0, pers1
  ret
  .cfi_endproc

## Test a relative relocation.
.globl hidden1
.hidden hidden1
hidden1:
  .cfi_startproc
  .cfi_personality 0, hidden1
  ret
  .cfi_endproc

#--- b.s
.set pers2, ext2+0x20
.text
.globl f2
f2:
  .cfi_startproc
  .cfi_personality 0, pers2
  ret
  .cfi_endproc

.globl hidden2
.hidden hidden2
hidden2:
  .cfi_startproc
  .cfi_personality 0, hidden2
  ret
  .cfi_endproc
