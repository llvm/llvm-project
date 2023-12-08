# REQUIRES: x86
## Test that input SHT_REL[A] retained by --emit-relocs are not matched by input section descriptions.

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: echo "SECTIONS { .rela.dyn : { *(.rela.data) } }" > %t.script
# RUN: ld.lld --hash-style=sysv -T %t.script --emit-relocs %t.o -o %t.so -shared
# RUN: llvm-readobj -r %t.so | FileCheck %s

## .rela.data is not listed, but don't error.
# RUN: echo 'SECTIONS { \
# RUN:   .dynsym : { *(.dynsym) } \
# RUN:   .gnu.hash : { *(.gnu.hash) } \
# RUN:   .hash : { *(.hash) } \
# RUN:   .dynstr : { *(.dynstr) } \
# RUN:   .dynamic : { *(.dynamic) } \
# RUN:   .rela.dyn : { *(.rela.dyn) } \
# RUN:   .text : { *(.text) } \
# RUN:   .data : { *(.data) } \
# RUN:   .comment : { *(.comment) } \
# RUN:   .symtab : { *(.symtab) } \
# RUN:   .shstrtab : { *(.shstrtab) } \
# RUN:   .strtab : { *(.strtab) } \
# RUN:  }' > %t1.script
# RUN: ld.lld -T %t1.script -shared --emit-relocs %t.o --orphan-handling=error -o /dev/null

.data
.quad .foo

# CHECK:      Relocations [
# CHECK-NEXT:   Section ({{.*}}) .rela.dyn {
# CHECK-NEXT:     0xF8 R_X86_64_64 .foo 0x0
# CHECK-NEXT:   }
# CHECK-NEXT:   Section ({{.*}}) .rela.data {
# CHECK-NEXT:     0xF8 R_X86_64_64 .foo 0x0
# CHECK-NEXT:   }
# CHECK-NEXT: ]
