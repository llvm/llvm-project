## Check that non-simple functions retained at their input addresses are not
## included in LongJmp's layout in non-relocation mode.

# REQUIRES: system-linux, asserts

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -nostdlib
# RUN: llvm-bolt %t.exe -o %t.bolt --lite=0 --print-cfg \
# RUN:   --print-only=retained --debug-only=longjmp > %t.log 2>&1
# RUN: FileCheck %s < %t.log

# CHECK: Binary Function "retained" after building cfg
# CHECK: IsSimple    : 0
# CHECK: BOLT-DEBUG: LongJmp layout: main fragment _start starts at 0x
# CHECK-NOT: BOLT-DEBUG: LongJmp layout: main fragment retained

  .text
  .globl _start
  .type _start, %function
_start:
  ret
  .size _start, .-_start

## The unknown indirect branch makes the function non-simple without removing
## its CFG or instructions. In non-relocation mode,
## shouldEmitFunctionFragment() accepts it, while BinaryContext::shouldEmit()
## rejects it.
  .globl retained
  .type retained, %function
retained:
  br x0
  ret
  .size retained, .-retained
