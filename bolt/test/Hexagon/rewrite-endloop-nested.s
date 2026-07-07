## Verify that BOLT can fully rewrite a binary containing nested hardware
## loops (loop0 inside loop1 with endloop0 and endloop1). Nested loops
## exercise the endloop1 code path in createBundle, which requires a
## minimum of 3 instructions in the endloop1 packet (compared to 2 for
## endloop0). BOLT must preserve both loop-end markers through the
## round-trip.

# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-linux-musl %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe --emit-relocs -e _start
# RUN: llvm-bolt %t.exe -o %t.bolt 2>&1 | FileCheck --check-prefix=BOLT %s
# RUN: llvm-objdump -d %t.bolt | FileCheck %s

# BOLT-NOT: BOLT-ERROR

# CHECK-LABEL: <test_nested_loop>:
# CHECK:       loop1(
# CHECK:       loop0(
# CHECK:       :endloop0
# CHECK:       :endloop1

  .text
  .globl _start
  .type _start,@function
  .p2align 4
_start:
  call test_nested_loop
  jumpr r31
  .size _start, .-_start

  .globl test_nested_loop
  .type test_nested_loop,@function
  .p2align 4
test_nested_loop:
  loop1(.Louter, #3)
.Louter:
  loop0(.Linner, #4)
.Linner:
  {
    r0 = add(r0, #1)
    nop
  }:endloop0
  {
    r1 = add(r1, #1)
    nop
    nop
  }:endloop1
  jumpr r31
  .size test_nested_loop, .-test_nested_loop
