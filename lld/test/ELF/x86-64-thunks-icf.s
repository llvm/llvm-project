# REQUIRES: x86
# Test that --icf=all interacts correctly with thunks. Two identical
# functions (foo, bar) are in the same region and get folded by ICF.
# A far-away caller (_start) references both; after folding, both
# resolve to foo and the thunk to foo is reused. foo itself calls
# target without needing a thunk (both are nearby).

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/main.s -o %t/main.o
# RUN: ld.lld --icf=all --print-icf-sections -T %t/script.lds %t/main.o -o %t/out 2>&1 | FileCheck --check-prefix=ICF %s
# RUN: llvm-objdump -d --no-show-raw-insn %t/out | FileCheck %s

# ICF: selected section {{.*}}:(.text.foo)
# ICF:   removing identical section {{.*}}:(.text.bar)

## foo calls target directly (no thunk, both near 0x10000).
# CHECK:      <foo>:
# CHECK-NEXT:   callq {{.*}} <target>
# CHECK-NEXT:   retq

# CHECK:      <target>:
# CHECK-NEXT:   retq

## _start is far away; both calls resolve to foo after ICF folding.
## A single thunk is reused for both calls.
# CHECK:      <_start>:
# CHECK-NEXT:   callq {{.*}} <__X86_64LongThunk_foo>
# CHECK-NEXT:   callq {{.*}} <__X86_64LongThunk_foo>
# CHECK-NEXT:   retq

# CHECK:      <__X86_64LongThunk_foo>:
# CHECK-NEXT:   movabsq
# CHECK-NEXT:   leaq
# CHECK-NEXT:   addq    %r10, %r11
# CHECK-NEXT:   jmpq    *%r11

#--- main.s
.section .text.foo,"ax",@progbits
.globl foo
.type foo, @function
foo:
  call target
  ret

.section .text.bar,"ax",@progbits
.globl bar
.type bar, @function
bar:
  call target
  ret

.section .text.target,"ax",@progbits
.globl target
.type target, @function
target:
  ret

.section .text._start,"ax",@progbits
.globl _start
.type _start, @function
_start:
  call foo
  call bar
  ret

#--- script.lds
SECTIONS {
    . = 0x10000;
    .text : { *(.text.foo) *(.text.bar) *(.text.target) }

    . = 0x200000000;
    .text.far : { *(.text._start) }
}
