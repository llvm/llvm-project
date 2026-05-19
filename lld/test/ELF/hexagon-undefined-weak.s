# REQUIRES: hexagon
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

## Undefined weak branch targets resolve to address zero.  Verify that
## no thunks are created and that branches encode without error.

.weak undefined_weak
.globl _start
.type _start, @function
_start:
  ## Simple call -- single-word packet.
  call undefined_weak

  ## Call in a two-word packet with an ALU op.
  { r0 = #0
    call undefined_weak }

  jumpr r31

## All branches target address zero.
# CHECK:       <_start>:
# CHECK-NEXT:  { call 0x0 <undefined_weak> }
# CHECK-NEXT:  { call 0x0 <undefined_weak>
# CHECK-NEXT:    r0 = #0x0 }
# CHECK-NEXT:  { jumpr r31 }
# CHECK-NOT:   __hexagon_thunk
