# REQUIRES: hexagon
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf main.s -o main.o
# RUN: ld.lld main.o -o test
# RUN: llvm-objdump -d --no-show-raw-insn test | FileCheck %s

## Test thunk range scenarios for Hexagon R_HEX_B22_PCREL relocations.
## R_HEX_B22_PCREL has a range of +/- 8MB (0x800000 bytes).

#--- main.s
.globl _start
.type _start, %function
_start:
  call target_within_range_max
  call target_beyond_range
  call target_within_range_min
  call target_beyond_range_min
  call target_multiple_calls
  call target_multiple_calls
  call target_close
  jumpr r31

target_close:
  jumpr r31

## Target at maximum positive range (8MB - 4 bytes from _start)
## We need to account for the instructions above: 7 calls + 1 jumpr = 8 * 4 = 32 bytes
.skip 0X7fffbc
.globl target_within_range_max
.type target_within_range_max, %function
target_within_range_max:
  jumpr r31

## Target just beyond maximum positive range (needs thunk)
.skip 8
.globl target_beyond_range
.type target_beyond_range, %function
target_beyond_range:
  call target_within_range_max
  jumpr r31

## Target for multiple calls test
.skip 0x100000
.globl target_multiple_calls
.type target_multiple_calls, %function
target_multiple_calls:
  jumpr r31

## Now place targets at maximum negative range
## We'll put these before _start in memory layout
.section .text_negative, "ax", %progbits

## Target at maximum negative range (-8MB + 4 bytes from _start)
.globl target_within_range_min
.type target_within_range_min, %function
target_within_range_min:
  call target_close
  jumpr r31

.skip 0X7ffff4

## Target beyond maximum negative range (needs thunk)
.globl target_beyond_range_min
.type target_beyond_range_min, %function
target_beyond_range_min:
  jumpr r31

## Verify thunk generation for targets beyond B22_PCREL range
# CHECK: 000200b4 <__hexagon_thunk_target_within_range_min_from_.text.thunk>:
# CHECK-NEXT:    200b4: { immext(#0x900000)
# CHECK-NEXT:    200b8:   jump 0x9200cc }

# CHECK: 000200bc <__hexagon_thunk_target_beyond_range_min_from_.text.thunk>:
# CHECK-NEXT:    200bc: { immext(#0x1100000)
# CHECK-NEXT:    200c0:   jump 0x11200c8 }

# CHECK: 000200c4 <__hexagon_thunk_target_multiple_calls_from_.text.thunk>:
# CHECK-NEXT:    200c4: { immext(#0x8fffc0)
# CHECK-NEXT:    200c8:   jump 0x9200c0 }

## Verify _start calls - some direct, some via thunks
# CHECK: 000200cc <_start>:
# CHECK-NEXT:    200cc: { call 0x8200ac }
# CHECK-NEXT:    200d0: { call 0x8200b8 }
# CHECK-NEXT:    200d4: { call 0x200b4 }
# CHECK-NEXT:    200d8: { call 0x200bc }
# CHECK-NEXT:    200dc: { call 0x200c4 }
# CHECK-NEXT:    200e0: { call 0x200c4 }
# CHECK-NEXT:    200e4: { call 0x200ec }

# CHECK: 000200ec <target_close>:
# CHECK-NEXT:    200ec: { jumpr r31 }

## Verify targets at maximum positive range (direct calls, no thunks needed)
# CHECK: 008200ac <target_within_range_max>:
# CHECK-NEXT:  8200ac: { jumpr r31 }

# CHECK: 008200b8 <target_beyond_range>:
# CHECK-NEXT:  8200b8: { call 0x8200ac }
# CHECK-NEXT:  8200bc: { jumpr r31 }

# CHECK: 009200c0 <target_multiple_calls>:
# CHECK-NEXT:  9200c0: { jumpr r31 }

## Verify targets in negative section and thunk for calling back to main section
# CHECK: 009200c4 <__hexagon_thunk__from_.text.thunk>:
# CHECK-NEXT:  9200c4: { immext(#0xff700000)
# CHECK-NEXT:  9200c8:   jump 0x200cc }

# CHECK: 009200cc <target_within_range_min>:
# CHECK-NEXT:  9200cc: { call 0x9200c4 }
# CHECK-NEXT:  9200d0: { jumpr r31 }

# CHECK: 011200c8 <target_beyond_range_min>:
# CHECK-NEXT: 11200c8: { jumpr r31 }
