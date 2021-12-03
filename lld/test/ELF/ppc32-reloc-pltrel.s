# REQUIRES: ppc

## Ensure R_PPC_PLTREL retains .got even in the absence of
## .got/_GLOBAL_OFFSET_TABLE_ references.

# RUN: llvm-mc -filetype=obj -triple=powerpc %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-readobj -Sdr %t.so | FileCheck %s

.section .got2,"aw",@progbits
.set .LTOC, .+0x8000

.text
.L0:
addis 30,30,.LTOC-.L0@ha
addi 30,30,.LTOC-.L0@l
bl baz+0x8000@plt

## DT_PPC_GOT must point to .got, which must have the 12-byte header.
## The only relocation is an R_PPC_JMP_SLOT.

# CHECK:      Sections [
# CHECK:        Name: .got (
# CHECK:        Address:
# CHECK-SAME:   {{ }}[[#%x,GOT:]]
# CHECK:        Size:
# CHECK-SAME:   {{ 12$}}
# CHECK:      DynamicSection [
# CHECK-NEXT:   Tag        Type     Name/Value
# CHECK:        0x70000000 PPC_GOT  [[#GOT]]
# CHECK:      Relocations [
# CHECK-NEXT:   Section ([[#]]) .rela.plt {
# CHECK-NEXT:     0x[[#%x,]] R_PPC_JMP_SLOT baz 0x0
# CHECK-NEXT:   }
# CHECK-NEXT: ]
