# REQUIRES: x86
## Show how symbols in GCed mergeable pieces behave.

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: ld.lld --gc-sections %t.o -o %t.elf
# RUN: llvm-readelf %t.elf --sections --syms | FileCheck %s

.section .rodata.merge,"aM",@progbits,4
a1: ## Unreferenced. In first fragment, kept by a2 reference.
    .short 1
a2: ## Referenced.
    .short 1
b1: ## Unreferenced. Discarded as second fragment is unreferenced.
    .short 1
b2: ## Unreferenced. Discarded as second fragment is unreferenced.
    .short 1
c1: ## Referenced.
    .short 1
c2: ## Unreferenced. In third fragment, kept by c1 reference.
    .short 1

.data
.global _start
_start:
    .quad a2
    .quad c1

# CHECK:      .rodata PROGBITS [[#%x, ADDR:]]

# CHECK:      Symbol table '.symtab' contains 6 entries:
# CHECK-NEXT:   Num:    Value          {{.*}} Ndx Name
# CHECK-NEXT:     0:                   {{.*}} UND{{ *$}}
# CHECK-NEXT:     1: {{0*}}[[#ADDR]]   {{.*}}     a1
# CHECK-NEXT:     2: {{0*}}[[#ADDR+2]] {{.*}}     a2
# CHECK-NEXT:     3: {{0*}}[[#ADDR]]   {{.*}}     c1
# CHECK-NEXT:     4: {{0*}}[[#ADDR+2]] {{.*}}     c2
# CHECK-NEXT:     5:                   {{.*}}     _start
