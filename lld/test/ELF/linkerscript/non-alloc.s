# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t

# RUN: echo "SECTIONS { .foo : {*(foo)} }" > %t.script
# RUN: ld.lld -o %t1 --script %t.script %t -shared
# RUN: llvm-readobj -elf-output-style=GNU -s -l %t1 | FileCheck %s

# Test that we create all necessary PT_LOAD. It is a harmless oddity that
# foo ends in a PT_LOAD. We use to stop at the first non-alloc, causing
# us to not create PT_LOAD for linker generated sections.

# CHECK: Program Headers:
# CHECK-NEXT:  Type
# CHECK-NEXT:  PHDR
# CHECK-NEXT:  LOAD {{.*}} R
# CHECK-NEXT:  LOAD {{.*}} R E
# CHECK-NEXT:  LOAD {{.*}} RW

# CHECK:      Section to Segment mapping:
# CHECK-NEXT:  Segment Sections...
# CHECK-NEXT:   00
# CHECK-NEXT:   01     .foo .dynsym .hash .dynstr
# CHECK-NEXT:   02     .text
# CHECK-NEXT:   03     .dynamic

nop
.section foo
.quad 0
