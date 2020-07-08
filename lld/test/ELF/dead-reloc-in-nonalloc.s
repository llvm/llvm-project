# REQUIRES: x86
## Test that -z dead-reloc-in-nonalloc= can customize the tombstone value we
## use for an absolute relocation referencing a discarded symbol.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld --icf=all -z dead-reloc-in-nonalloc=.debug_info=0xaaaaaaaa \
# RUN:   -z dead-reloc-in-nonalloc=.not_debug=0xbbbbbbbb %t.o -o %t
# RUN: llvm-objdump -s %t | FileCheck %s --check-prefixes=COMMON,AA
## 0xaaaaaaaa == 2863311530
# RUN: ld.lld --icf=all -z dead-reloc-in-nonalloc=.debug_info=2863311530 \
# RUN:   -z dead-reloc-in-nonalloc=.not_debug=0xbbbbbbbb %t.o -o - | cmp %t -

# COMMON:      Contents of section .debug_addr:
# COMMON-NEXT:  0000 [[ADDR:[0-9a-f]+]] 00000000 ffffffff ffffffff

# AA:          Contents of section .debug_info:
# AA-NEXT:      0000 [[ADDR]] 00000000 aaaaaaaa 00000000
# AA:          Contents of section .not_debug:
# AA-NEXT:      0000 bbbbbbbb

## Specifying zero can get a behavior similar to GNU ld.
# RUN: ld.lld --icf=all -z dead-reloc-in-nonalloc=.debug_info=0 %t.o -o %tzero
# RUN: llvm-objdump -s %tzero | FileCheck %s --check-prefixes=COMMON,ZERO

# ZERO:        Contents of section .debug_info:
# ZERO-NEXT:    0000 {{[0-9a-f]+}}000 00000000 00000000 00000000

## Glob works.
# RUN: ld.lld --icf=all -z dead-reloc-in-nonalloc='.debug_i*=0xaaaaaaaa' \
# RUN:   -z dead-reloc-in-nonalloc='[.]not_debug=0xbbbbbbbb' %t.o -o - | cmp %t -

## If a section matches multiple option. The last option wins.
# RUN: ld.lld --icf=all -z dead-reloc-in-nonalloc='.debug_info=1' \
# RUN:   -z dead-reloc-in-nonalloc='.debug_i*=0' %t.o -o - | cmp %tzero -

## Test all possible invalid cases.
# RUN: not ld.lld -z dead-reloc-in-nonalloc= 2>&1 | FileCheck %s --check-prefix=USAGE
# RUN: not ld.lld -z dead-reloc-in-nonalloc=a= 2>&1 | FileCheck %s --check-prefix=USAGE
# RUN: not ld.lld -z dead-reloc-in-nonalloc==0 2>&1 | FileCheck %s --check-prefix=USAGE

# USAGE: error: -z dead-reloc-in-nonalloc=: expected <section_glob>=<value>

# RUN: not ld.lld -z dead-reloc-in-nonalloc=a=-1 2>&1 | FileCheck %s --check-prefix=NON-INTEGER

# NON-INTEGER: error: -z dead-reloc-in-nonalloc=: expected a non-negative integer, but got '-1'

# RUN: not ld.lld -z dead-reloc-in-nonalloc='['=0 2>&1 | FileCheck %s --check-prefix=INVALID

# INVALID: error: -z dead-reloc-in-nonalloc=: invalid glob pattern: [

.globl _start
_start:
  ret

## .text.1 will be folded by ICF.
.section .text.1,"ax"
  ret

.section .debug_addr
  .quad .text+8
  .quad .text.1+8

.section .debug_info
  .quad .text+8
  .quad .text.1+8

## Test a non-.debug_ section.
.section .not_debug
  .long .text.1+8
