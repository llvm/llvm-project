# REQUIRES: hexagon
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf external.s -o external.o
# RUN: ld.lld -shared external.o -soname external.so -o external.so
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf main.s -o main.o
# RUN: ld.lld -T a.t main.o external.so -o a
# RUN: llvm-objdump -d --no-show-raw-insn a | FileCheck %s

## Test thunk range scenarios for Hexagon R_HEX_PLT_B22_PCREL relocations.
## PLT calls use the same +/- 8 MiB range as regular B22_PCREL calls.
## When the PLT entry is beyond this range, a thunk must be created.

## Out-of-range (9 MiB away): _start calls the thunk instead of the PLT entry directly.
# CHECK:      <_start>:
# CHECK-NEXT:   20000: { call 0x20008 <__hexagon_thunk_extern_func_from_.text.thunk> }
# CHECK-NEXT:          { jumpr r31 }

## The thunk uses immext+jump to reach the PLT entry.
# CHECK:      <__hexagon_thunk_extern_func_from_.text.thunk>:
# CHECK-NEXT:   20008: { immext(#0x900000)
# CHECK-NEXT:            jump 0x920020 <extern_func@plt> }

## Within range (2 MiB away): near calls the PLT entry directly (no thunk).
# CHECK:      <near>:
# CHECK-NEXT:  720000: { call 0x920020 <extern_func@plt> }
# CHECK-NEXT:          { jumpr r31 }

#--- external.s
.globl extern_func
.type extern_func, @function
extern_func:
  jumpr r31

#--- main.s
.globl _start
.type _start, @function
_start:
  call extern_func@PLT
  jumpr r31

.section .text_near,"ax",@progbits
.globl near
.type near, @function
near:
  call extern_func@PLT
  jumpr r31

#--- a.t
SECTIONS {
  .text 0x20000 : { *(.text) }
  .text_near 0x720000 : { *(.text_near) }
  .plt 0x920000 : { *(.plt) }
}
