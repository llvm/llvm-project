# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o

# RUN: %lld -o %t/test %t/test.o -order_file %t/order-file
# RUN: llvm-objdump --section-headers -d --no-show-raw-insn %t/test | FileCheck %s
# CHECK-LABEL: Sections:
# CHECK:       __cstring {{[^ ]*}} {{0*}}[[#%x, CSTRING_ADDR:]]
# CHECK-LABEL: Disassembly of section __TEXT,__text:
## L._str should end up at CSTRING_ADDR + 4, and leaq is 7 bytes long so we
## have RIP = ADDR + 7
# CHECK:       [[#%x, ADDR:]]: leaq
# CHECK-SAME:    [[#%u, CSTRING_ADDR + 4 - ADDR - 7]](%rip), %rsi {{.*}} <_bar_str+0x4>

# RUN: llvm-readobj --string-dump=__cstring %t/test | FileCheck %s --check-prefix=STRINGS
# STRINGS: bar
# STRINGS: Private symbol
# STRINGS: foo

#--- order-file
_bar_str
_foo_str

#--- test.s
.text
.globl _main, _foo_str, _bar_str

_main:
  leaq L_.str(%rip), %rsi
  mov $0, %rax
  ret

.section __TEXT,__cstring
_foo_str:
  .asciz "foo"

_bar_str:
  .asciz "bar"

## References to this generate a section relocation
## N.B.: ld64 doesn't actually reorder symbols in __cstring based on the order
##       file. Our implementation only does does so if --no-literal-merge is
##       specified. I'm not sure how else to test section relocations that
##       target an address inside a relocated symbol: using a non-__cstring
##       section would cause llvm-mc to emit a symbol relocation instead using
##       the nearest symbol. It might be more consistent for LLD to disable
##       symbol-based cstring reordering altogether and leave this functionality
##       untested, at least until we find a real-world use case...
L_.str:
  .asciz "Private symbol"

.subsections_via_symbols
