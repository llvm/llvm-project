## Verify that BOLT can rewrite double-register (register pair) operations.
## Hexagon uses register pairs (r1:0, r3:2, etc.) for 64-bit operations
## including double-word loads, stores, and combine instructions.

# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-linux-musl %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe --emit-relocs -e _start
# RUN: llvm-bolt %t.exe -o %t.bolt 2>&1 | FileCheck --check-prefix=BOLT %s
# RUN: llvm-objdump -d %t.bolt | FileCheck %s

# BOLT-NOT: BOLT-ERROR

  .text
  .globl _start
  .type _start,@function
  .p2align 4
_start:
  call test_combine
  call test_memd
  jumpr r31
  .size _start, .-_start

##============================================================================
## Register pair combine operations.
##============================================================================
# CHECK-LABEL: <test_combine>:
# CHECK:       r1:0 = combine(r2,r3)
# CHECK:       r5:4 = combine(#0x1,#0x0)

  .globl test_combine
  .type test_combine,@function
  .p2align 4
test_combine:
  r1:0 = combine(r2, r3)
  r5:4 = combine(#1, #0)
  jumpr r31
  .size test_combine, .-test_combine

##============================================================================
## Double-word load and store (memd) with register pairs.
##============================================================================
# CHECK-LABEL: <test_memd>:
# CHECK:       r1:0 = memd(r2+#0x0)
# CHECK:       memd(r3+#0x0) = r1:0

  .globl test_memd
  .type test_memd,@function
  .p2align 4
test_memd:
  r1:0 = memd(r2 + #0)
  memd(r3 + #0) = r1:0
  jumpr r31
  .size test_memd, .-test_memd
