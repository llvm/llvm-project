# REQUIRES: x86
# RUN: rm -rf %t && mkdir -p %t
# RUN: split-file %s %t && cd %t

## Test that lld's orphan section placement can handle a relocatable link where
## the relocation section is seen before the relocated section. 

## Create a relocatable object with the relocations before the relocated section
# RUN: llvm-mc -filetype=obj -triple=x86_64 foo.s -o foo.o
# RUN: ld.lld -r  foo.o -T script.ld -o foo_mc.o

## Rename the sections to make them orphans
# RUN: llvm-objcopy \
# RUN: --rename-section .text=.com.text \
# RUN: --rename-section .rela.text=.rela.com.text \
# RUN: foo_mc.o foo_mc.o

# RUN: ld.lld -r foo_mc.o  -T script.ld -o foo_mc_after.o
# RUN:  llvm-readelf -S foo_mc_after.o | FileCheck %s
# CHECK:       .com.text         PROGBITS        0000000000000000 000040 000007 00  AX  0   0 16
# CHECK-NEXT:  .rela.com.text    RELA            0000000000000000 000048 000018 18   I  4   1  8

#--- foo.s
  .text
  .globl	foo
  .p2align	4
  .type	foo,@function
foo:
  mov $bar, %rax

#--- script.ld
SECTIONS
{
  .rela.text    0 : { *(.rela.text) }
  .text         0 : { *(.text) }
}

