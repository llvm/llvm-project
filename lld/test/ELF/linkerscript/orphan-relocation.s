# REQUIRES: x86
## Test that orphan section placement can handle a relocatable link where
## the relocation section is seen before the relocated section.

# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
## In a.ro, .rela.text precedes its relocated section.
# RUN: ld.lld -r a.o -T 1.lds -o a.ro
# RUN: llvm-readelf -S a.ro | FileCheck %s
# CHECK:       .rela.text    RELA
# CHECK-NEXT:  .text         PROGBITS

# RUN: llvm-objcopy --rename-section .text=.com.text --rename-section .rela.text=.rela.com.text a.ro a1.o

## Regression test for #156354 , where we added an orphan RELA section before its relocated section.
# RUN: ld.lld -r a1.o -o a1.ro
# RUN: llvm-readelf -S a1.ro | FileCheck %s --check-prefix=CHECK1
# CHECK1:       .com.text         PROGBITS
# CHECK1-NEXT:  .rela.com.text    RELA

#--- a.s
.globl foo
foo:
  call foo

#--- 1.lds
SECTIONS {
  .rela.text 0 : { *(.rela.text) }
  .text      0 : { *(.text) }
}
