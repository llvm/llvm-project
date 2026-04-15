## Verify that BOLT can build a CFG for functions with indirect register
## calls (callr). isIndirectCall identifies J2_callr and hasPCRelOperand
## returns false. CFG building must not attempt to resolve a nonexistent
## branch target symbol.

# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-linux-musl %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe --emit-relocs -e _start
# RUN: llvm-bolt %t.exe --print-cfg --print-only=test_indirect_call \
# RUN:   -o /dev/null > %t.log 2>&1
# RUN: FileCheck %s --input-file=%t.log

# CHECK: Binary Function "test_indirect_call" after building cfg
# CHECK: callr r0

  .text
  .globl _start
  .type _start,@function
  .p2align 4
_start:
    call test_indirect_call
    jumpr r31
  .size _start, .-_start

  .globl test_indirect_call
  .type test_indirect_call,@function
  .p2align 4
test_indirect_call:
    callr r0
    jumpr r31
  .size test_indirect_call, .-test_indirect_call
