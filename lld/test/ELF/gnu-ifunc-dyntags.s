# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: ld.lld -pie %t.o -o %tout
# RUN: llvm-objdump --section-headers %tout | FileCheck %s
# RUN: llvm-readobj --dynamic-table -r %tout | FileCheck %s --check-prefix=TAGS

## Check we produce DT_PLTREL/DT_JMPREL/DT_PLTGOT and DT_PLTRELSZ tags
## when there are no other relocations except R_*_IRELATIVE.

# CHECK:  Name          Size   VMA
# CHECK:  .rela.dyn   00000030 0000000000000248
# CHECK:  .got.plt    00000010 0000000000003370

# TAGS:   Tag                Type                 Name/Value
# TAGS:   0x0000000000000007 RELA                 0x248
# TAGS:   0x0000000000000008 RELASZ               48 (bytes)
# TAGS-NOT: JMPREL
# TAGS-NOT: PLTREL

# TAGS:      Relocations [
# TAGS-NEXT:   Section {{.*}} .rela.dyn {
# TAGS-NEXT:     R_X86_64_IRELATIVE
# TAGS-NEXT:     R_X86_64_IRELATIVE
# TAGS-NEXT:   }
# TAGS-NEXT: ]

.text
.type foo STT_GNU_IFUNC
.globl foo
foo:
 ret

.type bar STT_GNU_IFUNC
.globl bar
bar:
 ret

.globl _start
_start:
 call foo
 call bar
