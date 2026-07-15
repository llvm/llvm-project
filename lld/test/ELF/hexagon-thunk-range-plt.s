# REQUIRES: hexagon
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf external.s -o external.o
# RUN: ld.lld -shared external.o -soname external.so -o external.so

## PLT calls within range (2 MiB padding) — no thunks needed.
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf main.s -o main.o
# RUN: ld.lld main.o external.so -o test
# RUN: llvm-objdump -d --no-show-raw-insn test | \
# RUN:     FileCheck --check-prefix=INRANGE %s

## PLT calls out of range (>8 MiB padding) — thunks required.
# RUN: llvm-mc -filetype=obj \
# RUN:         -triple=hexagon-unknown-elf main-large.s -o main-large.o
# RUN: ld.lld main-large.o external.so -o test-large
# RUN: llvm-objdump -d --no-show-raw-insn test-large | \
# RUN:     FileCheck --check-prefix=OUTRANGE %s

## Test thunk range scenarios for Hexagon R_HEX_PLT_B22_PCREL relocations.
## PLT calls use the same +/- 8 MiB range as regular B22_PCREL calls.
## When the PLT entry is beyond this range, a thunk must be created.

#--- external.s
.globl extern_func
.type extern_func, @function
extern_func:
  jumpr r31

#--- main.s
## Within-range case: 2 MiB padding keeps PLT entries reachable.
.globl _start
.type _start, @function
_start:
  call extern_func@PLT
  jumpr r31

.skip 0x200000

#--- main-large.s
## Out-of-range case: >8 MiB padding pushes PLT entries beyond B22_PCREL range.
.globl _start
.type _start, @function
_start:
  call extern_func@PLT
  jumpr r31

.skip 0x900000

## Within-range: _start calls the PLT entry directly (no thunk).
# INRANGE:      <_start>:
# INRANGE-NEXT:   201ac: { call 0x2201e0 <extern_func@plt> }
# INRANGE-NEXT:          { jumpr r31 }

## Out-of-range: thunk uses immext+jump to reach the PLT entry.
# OUTRANGE:      <__hexagon_thunk_extern_func_from_.text.thunk>:
# OUTRANGE-NEXT:   201ac: { immext(#0x900000)
# OUTRANGE-NEXT:            jump 0x9201e0 <extern_func@plt> }

## _start calls the thunk instead of the PLT entry directly.
# OUTRANGE:      <_start>:
# OUTRANGE-NEXT:   201b4: { call 0x201ac <__hexagon_thunk_extern_func_from_.text.thunk> }
# OUTRANGE-NEXT:          { jumpr r31 }
