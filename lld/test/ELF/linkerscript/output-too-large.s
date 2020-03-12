# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=i686 %s -o %t1.o
# RUN: echo "SECTIONS { .text : { . = 0xffffffff; *(.text*); } }" > %t1.script
# RUN: not ld.lld --no-check-sections -T %t1.script %t1.o -o /dev/null 2>&1 | FileCheck %s

## Error if an address is greater than or equal to 2**32 for ELF32.
## When -M is specified, print the link map even if such an error occurs,
## because the link map can help diagnose problems.
# RUN: not ld.lld -T %t1.script %t1.o -M -o /dev/null 2>&1 | \
# RUN:   FileCheck --check-prefix=MAP1 %s

# MAP1:           VMA      LMA     Size Align Out     In      Symbol
# MAP1-NEXT:        0        0 100000001     4 .text
# MAP1-NEXT:        0        0 ffffffff     1         . = 0xffffffff
# MAP1-NEXT: 100000000 100000000        1     4         {{.*}}.o:(.text)
# MAP1:      error: section .text at 0x0 of size 0x100000001 exceeds available address space

## Error if an address is greater than or equal to 2**63.
## Print a link map if -M is specified.
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t2.o
# RUN: echo "SECTIONS { .text : { . = 0x8fffffffffffffff; *(.text*); } }" > %t2.script
# RUN: not ld.lld -T %t2.script -M %t2.o -o /dev/null 2>&1 | \
# RUN:   FileCheck --check-prefixes=MAP2,CHECK %s

# MAP2:                   VMA              LMA     Size Align Out     In      Symbol
# MAP2:      9000000000000000 9000000000000000        1     4         {{.*}}.o:(.text)
# MAP2-NEXT: 9000000000000000 9000000000000000        0     1                 _start

# CHECK: error: output file too large

.global _start
_start:
  nop
