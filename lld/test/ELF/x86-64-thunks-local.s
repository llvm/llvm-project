# REQUIRES: x86
# Test x86-64 range extension thunks for calls to local symbols.
#
# When a call targets a local symbol, the assembler generates a relocation
# against the STT_SECTION symbol with an addend encoding the function offset
# (plus the -4 PC bias for x86-64). This test verifies that after redirecting
# through a thunk:
# 1. The thunk correctly jumps to the local symbol's address (section + offset)
# 2. The call relocation to the thunk has the correct addend (-4 for x86-64)
# 3. The thunk name is properly disambiguated for section symbols

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/main.s -o %t/main.o
# RUN: ld.lld %t/main.o -T %t/script.lds -o %t/out
# RUN: llvm-objdump -d --no-show-raw-insn %t/out | FileCheck %s

#--- main.s
.text

## _start calls a local symbol in a far-away section.
## The assembler encodes this as R_X86_64_PLT32 against the STT_SECTION symbol
## for .text.far with addend = offset_of_far_local - 4 = 0x100 - 4 = 0xFC.
.globl _start
.type _start, @function
_start:
    call far_local
    ret

## The call should reach the thunk entry point exactly (no +0xN offset).
## The thunk name includes the actual destination offset (0x100) since the
## destination is a nameless STT_SECTION symbol.
# CHECK-LABEL: <_start>:
# CHECK-NEXT:    callq {{.*}} <__X86_64LongThunk__100>
# CHECK-NEXT:    retq

## Padding to push _start out of range of the far section.
.section .text.pad,"ax",@progbits
nop

## far_local is a local (non-global) function placed at 8GiB via linker script.
## Because it is local, the assembler will generate a relocation against the
## STT_SECTION symbol for .text.far with an addend.
.section .text.far,"ax",@progbits
## Add some padding before the local function so the addend is non-trivial.
.space 0x100
far_local:
    ret

## The thunk must jump to far_local's actual address (8GiB + 0x100),
## NOT to the start of .text.far (8GiB + 0x0).
# CHECK-LABEL: <__X86_64LongThunk__100>:
# CHECK-NEXT:    movabsq
# CHECK-NEXT:    leaq
# CHECK-NEXT:    addq    %r10, %r11
# CHECK-NEXT:    jmpq    *%r11

## Verify that far_local is at the expected address.
# CHECK-LABEL: <far_local>:
# CHECK-NEXT:    retq

#--- script.lds
SECTIONS {
    . = 0x10000;
    .text : { *(.text) }

    . = 0x80000000;
    .text.pad : { *(.text.pad) }

    ## Place far section at 8GiB
    . = 0x200000000;
    .text.far : { *(.text.far) }
}
