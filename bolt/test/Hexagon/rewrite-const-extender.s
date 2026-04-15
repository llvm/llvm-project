## Verify that BOLT correctly handles constant extender (immext / A4_ext)
## instructions. During disassembly BOLT sees A4_ext as a real instruction.
## During emission, createBundle skips it because the MC code emitter
## auto-regenerates immext from ## operands. If the skip logic breaks,
## packets exceed 4 slots or double-extend.

# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-linux-musl %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe --emit-relocs -e _start
# RUN: llvm-bolt %t.exe -o %t.bolt 2>&1 | FileCheck --check-prefix=BOLT %s
# RUN: llvm-objdump -d %t.bolt | FileCheck %s

# BOLT-NOT: BOLT-ERROR

# CHECK-LABEL: <test_immext>:
# CHECK:       immext(
# CHECK:       r0 = add(r1,##0x10000)

  .text
  .globl _start
  .type _start,@function
  .p2align 4
_start:
    call test_immext
    jumpr r31
  .size _start, .-_start

  .globl test_immext
  .type test_immext,@function
  .p2align 4
test_immext:
    r0 = add(r1, ##65536)
    jumpr r31
  .size test_immext, .-test_immext
