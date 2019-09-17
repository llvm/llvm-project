# REQUIRES: x86

## On RELA targets, --apply-dynamic-relocs writes addends to the relocated positions.

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t.so -shared --apply-dynamic-relocs
# RUN: llvm-readobj -r -S -l --section-data %t.so | FileCheck -check-prefixes=CHECK,APPLY %s

# RUN: ld.lld %t.o -o %t2.so -shared
# RUN: llvm-readobj -r -S -l --section-data %t2.so | FileCheck -check-prefixes=CHECK,NOAPPLY %s
# RUN: ld.lld %t.o -o %t3.so -shared --no-apply-dynamic-relocs
# RUN: cmp %t2.so %t3.so

# CHECK:        Name: .got
# CHECK:        Address: 0x[[GOT:.*]]
# CHECK:        SectionData (
# APPLY-NEXT:     0000: 30220000 00000000                |
# NOAPPLY-NEXT:   0000: 00000000 00000000                |
# CHECK-NEXT:   )

# CHECK:      Relocations [
# CHECK-NEXT:   Section ({{.*}}) .rela.dyn {
# CHECK-NEXT:     0x[[GOT]] R_X86_64_RELATIVE - 0x[[ADDEND:.*]]
# CHECK-NEXT:   }
# CHECK-NEXT: ]

# CHECK:      Type: PT_DYNAMIC
# CHECK-NEXT: Offset: 0x230
# CHECK-NEXT: VirtualAddress: 0x[[ADDEND]]
# CHECK-NEXT: PhysicalAddress: 0x[[ADDEND]]

cmpq    $0, _DYNAMIC@GOTPCREL(%rip)
