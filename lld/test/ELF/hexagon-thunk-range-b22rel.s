# REQUIRES: hexagon
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf main.s -o main.o
# RUN: ld.lld -T lds main.o -o test
# RUN: llvm-objdump -d --no-show-raw-insn test | FileCheck %s

## Test thunk range scenarios for Hexagon R_HEX_B22_PCREL relocations.
## R_HEX_B22_PCREL has a range of +/- 8MB (0x800000 bytes).

#--- main.s
.globl _start
.type _start, %function
_start:
  call target_within_range_max
  call target_beyond_range
  call target_beyond_range_min
  call target_within_range_min
  call target_multiple_calls
  call target_multiple_calls
  call target_close
  jumpr r31

target_close:
  jumpr r31

## Target at maximum positive range (8MB - 4 bytes from _start)
.section .text_max,"ax",@progbits
.globl target_within_range_max
.type target_within_range_max, %function
target_within_range_max:
  jumpr r31

## Target just beyond maximum positive range (needs thunk)
.skip 4
.globl target_beyond_range
.type target_beyond_range, %function
target_beyond_range:
  call target_within_range_max
  jumpr r31

## Target for multiple calls test
.section .text_multiple,"ax",@progbits
.globl target_multiple_calls
.type target_multiple_calls, %function
target_multiple_calls:
  jumpr r31

## Now place targets at maximum negative range
## We'll put these before _start in memory layout
.section .text_negative,"ax",@progbits

## Target beyond maximum negative range (needs thunk)
.globl target_beyond_range_min
.type target_beyond_range_min, %function
target_beyond_range_min:
  jumpr r31

## Target at maximum negative range (-8MB from the call at _start+12)
.skip 4
.globl target_within_range_min
.type target_within_range_min, %function
target_within_range_min:
  call target_close
  jumpr r31

#--- lds
SECTIONS {
  .text_negative 0x800004 : { *(.text_negative) }
  .text 0x1000000 : { *(.text) }
  .text_max 0x17ffffc : { *(.text_max) }
  .text_multiple 0x1900000 : { *(.text_multiple) }
}

## Verify targets in negative section and thunk for calling back to main section
# CHECK:      <target_beyond_range_min>:
# CHECK-NEXT:  800004: { jumpr r31 }

# CHECK:      <target_within_range_min>:
# CHECK-NEXT:  80000c: { call 0x800014 <__hexagon_thunk__from_.text.thunk> }
# CHECK-NEXT:          { jumpr r31 }

# CHECK:      <__hexagon_thunk__from_.text.thunk>:
# CHECK-NEXT:  800014: { immext(#0x7fffc0)
# CHECK-NEXT:            jump 0x1000000 <_start> }

## Verify _start calls - some direct, some via thunks
# CHECK:      <_start>:
# CHECK-NEXT: 1000000: { call 0x17ffffc <target_within_range_max> }
# CHECK-NEXT:          { call 0x1000034 <__hexagon_thunk_target_beyond_range_from_.text.thunk> }
# CHECK-NEXT:          { call 0x1000024 <__hexagon_thunk_target_beyond_range_min_from_.text.thunk> }
# CHECK-NEXT:          { call 0x80000c <target_within_range_min> }
# CHECK-NEXT:          { call 0x100002c <__hexagon_thunk_target_multiple_calls_from_.text.thunk> }
# CHECK-NEXT:          { call 0x100002c <__hexagon_thunk_target_multiple_calls_from_.text.thunk> }
# CHECK-NEXT:          { call 0x1000020 <target_close> }

# CHECK:      <target_close>:
# CHECK-NEXT: 1000020: { jumpr r31 }

## Verify thunk generation for targets beyond B22_PCREL range
# CHECK:      <__hexagon_thunk_target_beyond_range_min_from_.text.thunk>:
# CHECK-NEXT: 1000024: { immext(#0xff7fffc0)
# CHECK-NEXT:            jump 0x800004 <target_beyond_range_min> }

# CHECK:      <__hexagon_thunk_target_multiple_calls_from_.text.thunk>:
# CHECK-NEXT: 100002c: { immext(#0x8fffc0)
# CHECK-NEXT:            jump 0x1900000 <target_multiple_calls> }

# CHECK:      <__hexagon_thunk_target_beyond_range_from_.text.thunk>:
# CHECK-NEXT: 1000034: { immext(#0x7fffc0)
# CHECK-NEXT:            jump 0x1800004 <target_beyond_range> }

## Verify targets at maximum positive range (direct calls, no thunks needed)
# CHECK:      <target_within_range_max>:
# CHECK-NEXT: 17ffffc: { jumpr r31 }

# CHECK:      <target_beyond_range>:
# CHECK-NEXT: 1800004: { call 0x17ffffc <target_within_range_max> }
# CHECK-NEXT:          { jumpr r31 }

# CHECK:      <target_multiple_calls>:
# CHECK-NEXT: 1900000: { jumpr r31 }
