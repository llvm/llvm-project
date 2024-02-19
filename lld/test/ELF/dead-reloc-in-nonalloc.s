# REQUIRES: x86
## Test an absolute relocation referencing an undefined or DSO symbol, relocating
## a non-SHF_ALLOC section. Also test -z dead-reloc-in-nonalloc=.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo '.globl bar; bar = 42' | llvm-mc -filetype=obj -triple=x86_64 - -o %tabs.o
# RUN: ld.lld --gc-sections -z dead-reloc-in-nonalloc=.debug_info=0xaaaaaaaa \
# RUN:   -z dead-reloc-in-nonalloc=.not_debug=0xbbbbbbbb %t.o %tabs.o -o %t
# RUN: llvm-objdump -s %t | FileCheck %s --check-prefixes=COMMON,AA
## 0xaaaaaaaa == 2863311530
# RUN: ld.lld --gc-sections -z dead-reloc-in-nonalloc=.debug_info=2863311530 \
# RUN:   -z dead-reloc-in-nonalloc=.not_debug=0xbbbbbbbb %t.o %tabs.o -o - | cmp %t -

# COMMON:      Contents of section .debug_addr:
# COMMON-NEXT:  0000 [[ADDR:[0-9a-f]+]] 00000000 00000000 00000000

# AA:          Contents of section .debug_info:
# AA-NEXT:      0000 [[ADDR]] 00000000 aaaaaaaa 00000000
# AA:          Contents of section .not_debug:
# AA-NEXT:      0000 bbbbbbbb 2a000000 00000000          .

## Specifying zero can get a behavior similar to GNU ld.
# RUN: ld.lld --icf=all -z dead-reloc-in-nonalloc=.debug_info=0 %t.o %tabs.o -o %tzero
# RUN: llvm-objdump -s %tzero | FileCheck %s --check-prefixes=COMMON,ZERO

# ZERO:        Contents of section .debug_info:
# ZERO-NEXT:    0000 {{[0-9a-f]+}}000 00000000 00000000 00000000

## Glob works.
# RUN: ld.lld --gc-sections -z dead-reloc-in-nonalloc='.debug_i*=0xaaaaaaaa' \
# RUN:   -z dead-reloc-in-nonalloc='[.]not_debug=0xbbbbbbbb' %t.o %tabs.o -o - | cmp %t -

## If a section matches multiple option. The last option wins.
# RUN: ld.lld --icf=all -z dead-reloc-in-nonalloc='.debug_info=1' \
# RUN:   -z dead-reloc-in-nonalloc='.debug_i*=0' %t.o %tabs.o -o - | cmp %tzero -

# RUN: llvm-mc -filetype=obj -triple=x86_64 %S/Inputs/shared.s -o %t1.o
# RUN: ld.lld -shared -soname=t1.so %t1.o -o %t1.so
# RUN: ld.lld --gc-sections %t.o %t1.so -o %tso
# RUN: llvm-objdump -s %tso | FileCheck %s --check-prefix=SHARED

# SHARED:      Contents of section .not_debug:
# SHARED-NEXT: 0000 08000000 00000000 00000000           .

## Test all possible invalid cases.
# RUN: not ld.lld -z dead-reloc-in-nonalloc= 2>&1 | FileCheck %s --check-prefix=USAGE
# RUN: not ld.lld -z dead-reloc-in-nonalloc=a= 2>&1 | FileCheck %s --check-prefix=USAGE
# RUN: not ld.lld -z dead-reloc-in-nonalloc==0 2>&1 | FileCheck %s --check-prefix=USAGE

# USAGE: error: -z dead-reloc-in-nonalloc=: expected <section_glob>=<value>

# RUN: not ld.lld -z dead-reloc-in-nonalloc=a=-1 2>&1 | FileCheck %s --check-prefix=NON-INTEGER

# NON-INTEGER: error: -z dead-reloc-in-nonalloc=: expected a non-negative integer, but got '-1'

# RUN: not ld.lld -z dead-reloc-in-nonalloc='['=0 2>&1 | FileCheck %s --check-prefix=INVALID

# INVALID: error: -z dead-reloc-in-nonalloc=: invalid glob pattern, unmatched '[': [

.globl _start
_start:
  ret

## .text.1 will be folded by ICF or discarded by --gc-sections.
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

  .quad bar
