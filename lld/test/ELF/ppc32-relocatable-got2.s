# REQUIRES: ppc
## Test addend adjustment of R_PPC_PLTREL24 when copying relocations.
## If r_addend indicates .got2, adjust it by the local .got2's output section offset.

# RUN: llvm-mc -filetype=obj -triple=powerpc %s -o %t.o
# RUN: ld.lld -r %t.o %t.o -o %t
# RUN: llvm-readobj -r %t | FileCheck %s

# RUN: ld.lld -shared --emit-relocs %t.o %t.o -o %t.so
# RUN: llvm-readobj -r %t.so | FileCheck %s

# CHECK:      .rela.adjust {
# CHECK-NEXT:   R_PPC_REL16_HA .got2 0x8002
# CHECK-NEXT:   R_PPC_REL16_LO .got2 0x8006
# CHECK-NEXT:   R_PPC_PLTREL24 foo 0x8000
# CHECK-NEXT:   R_PPC_PLTREL24 bar 0x8000
# CHECK-NEXT:   R_PPC_REL16_HA .got2 0x8006
# CHECK-NEXT:   R_PPC_REL16_LO .got2 0x800A
# CHECK-NEXT:   R_PPC_PLTREL24 foo 0x8004
# CHECK-NEXT:   R_PPC_PLTREL24 bar 0x8004
# CHECK-NEXT: }
# CHECK-NEXT: .rela.no_adjust {
# CHECK-NEXT:   R_PPC_PLTREL24 foo 0x0
# CHECK-NEXT:   R_PPC_PLTREL24 foo 0x0
# CHECK-NEXT: }
.section .got2,"aw"
.long 0

.section .adjust,"ax"
bcl 20,30,.L0
.L0:
addis 30,30,.got2+0x8000-.L0@ha
addi 30,30,.got2+0x8000-.L0@l

## Refers to .got2+addend, adjust.
bl foo+0x8000@plt
bl bar+0x8000@plt

.section .no_adjust,"ax"
## Refers to .got, no adjustment.
bl foo@plt
