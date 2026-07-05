# REQUIRES: x86

## .data.foo and .data.bar are combined into .data,
## so their relocation sections should also be combined.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld --emit-relocs %t.o -o %t.so -shared
# RUN: llvm-readobj -r %t.so | FileCheck %s

# CHECK:       Relocations [
# CHECK-NEXT:    Section ({{.*}}) .rela.dyn {
# CHECK-NEXT:     [[ADDR1:[0-9a-f]+]] R_X86_64_64 zed 0x0
# CHECK-NEXT:     [[ADDR2:[0-9a-f]+]] R_X86_64_64 zed 0x0
# CHECK-NEXT:   }
# CHECK-NEXT:   Section ({{.*}}) .rela.data {
# CHECK-NEXT:     [[ADDR1]] R_X86_64_64 zed 0x0
# CHECK-NEXT:     [[ADDR2]] R_X86_64_64 zed 0x0
# CHECK-NEXT:   }
# CHECK-NEXT: ]

.section        .data.foo,"aw",%progbits
.quad zed
.section        .data.bar,"aw",%progbits
.quad zed
