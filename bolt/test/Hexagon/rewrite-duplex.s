## Verify that BOLT can fully rewrite a binary containing duplex
## instructions through the complete pipeline (disassembly -> CFG ->
## emit -> encode). The existing cfg-duplex.s only verifies CFG
## construction; this test verifies that the duplex instructions
## survive the full round-trip and produce correct output.
##
## Hexagon duplex instructions pack two sub-instructions into a single
## 32-bit word (parse bits = 0b00). When the assembler packs eligible
## pairs, the encoder creates duplex words that the disassembler
## splits back into sub-instruction MCInst operands.

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
  call test_duplex
  // Duplex-eligible pair: small immediate load + return.
  {
    r0 = #0
    jumpr r31
  }
  .size _start, .-_start

##============================================================================
## Function with multiple duplex-eligible instruction pairs.
## Small immediate assignments and register transfers can pack as duplexes.
##============================================================================
# CHECK-LABEL: <test_duplex>:
# CHECK:       r0 = #0x1
# CHECK:       r1 = #0x2
# CHECK:       r0 = add(r0,r1)
# CHECK:       jumpr r31

  .globl test_duplex
  .type test_duplex,@function
  .p2align 4
test_duplex:
  {
    r0 = #1
    r1 = #2
  }
  r0 = add(r0, r1)
  jumpr r31
  .size test_duplex, .-test_duplex
