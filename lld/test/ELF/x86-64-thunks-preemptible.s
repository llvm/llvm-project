# REQUIRES: x86
# Test that thunks for preemptible symbols correctly target the PLT entry
# rather than the symbol's definition address. In a shared library,
# calls to preemptible symbols must go through the PLT so that symbol
# interposition works at runtime.

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/main.s -o %t/main.o
# RUN: ld.lld -shared -T %t/script.lds %t/main.o -o %t/out.so
# RUN: llvm-objdump -d --no-show-raw-insn %t/out.so | FileCheck %s

## In a shared library, preemptible is not hidden so calls go through PLT.
## The PLT is near .text_low, but the caller is at 8GiB, so it needs a
## thunk. The thunk should jump to the PLT entry, not the definition.
# CHECK:      <caller>:
# CHECK-NEXT:   callq {{.*}} <__X86_64LongThunk_preemptible>
# CHECK-NEXT:   retq

## The thunk targets the PLT entry for the preemptible symbol.
# CHECK:      <__X86_64LongThunk_preemptible>:
# CHECK-NEXT:   movabsq
# CHECK-NEXT:   leaq
# CHECK-NEXT:   addq    %r10, %r11
# CHECK-NEXT:   jmpq    *%r11

#--- main.s
.section .text_low,"ax",@progbits
.globl preemptible
.type preemptible, @function
preemptible:
  ret

.section .text_high,"ax",@progbits
.globl caller
.type caller, @function
caller:
  call preemptible
  ret

#--- script.lds
SECTIONS {
    . = 0x10000;
    .text_low : { *(.text_low) }
    .plt : { *(.plt) *(.plt.*) }

    . = 0x200000000;
    .text_high : { *(.text_high) }
}
