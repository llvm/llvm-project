## Verify that BOLT handles conditional branches to external functions
## without crashing. convertJmpToTailCall must return false for
## conditional branches so they are treated as conditional tail calls.

# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-linux-musl %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe --emit-relocs -e _start
# RUN: llvm-bolt %t.exe --print-cfg --print-only=test_cond_tc -o /dev/null \
# RUN:   > %t.log 2>&1
# RUN: FileCheck %s --input-file=%t.log

# CHECK: Binary Function "test_cond_tc" after building cfg
# CHECK: if (p0) jump:nt
# CHECK: jump external_func # TAILCALL

  .text
  .globl _start
  .type _start,@function
  .p2align 4
_start:
    call test_cond_tc
    jumpr r31
  .size _start, .-_start

  .globl test_cond_tc
  .type test_cond_tc,@function
  .p2align 4
test_cond_tc:
    p0 = cmp.eq(r0, #0)
    if (p0) jump external_func
    jumpr r31
  .size test_cond_tc, .-test_cond_tc

  .globl external_func
  .type external_func,@function
  .p2align 4
external_func:
    jumpr r31
  .size external_func, .-external_func
