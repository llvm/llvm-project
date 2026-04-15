## Verify that BOLT can rewrite packets containing two constant extenders.
## A packet with two extended immediates requires two immext (A4_ext)
## instructions, totaling 4 words (the maximum packet size). The MC code
## emitter auto-generates immext when encoding extended immediates, so
## BOLT's createBundle must properly include both A4_ext instructions
## from the disassembly.

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
  call test_double_ext
  jumpr r31
  .size _start, .-_start

##============================================================================
## Two extended immediates in a single packet: each generates an immext.
## The packet occupies 4 words: immext + tfrsi + immext + tfrsi.
##============================================================================
# CHECK-LABEL: <test_double_ext>:
# CHECK:       immext
# CHECK:       r0 = ##
# CHECK:       immext
# CHECK:       r1 = ##

  .globl test_double_ext
  .type test_double_ext,@function
  .p2align 4
test_double_ext:
  {
    r0 = ##1234567
    r1 = ##7654321
  }
  jumpr r31
  .size test_double_ext, .-test_double_ext
