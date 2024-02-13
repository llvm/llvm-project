# REQUIRES: systemz
## Verify that R_390_GOTENT optimization is not performed on misaligned symbols.

# RUN: llvm-mc -filetype=obj -relax-relocations -triple=s390x-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t1
# RUN: llvm-readelf -S -r -x .got -x .got.plt %t1 | FileCheck --check-prefixes=CHECK %s
# RUN: llvm-objdump --no-print-imm-hex -d %t1 | FileCheck --check-prefix=DISASM %s

## We retain one .got entry for the unaligned symbol.
# CHECK:      Name              Type            Address          Off    Size   ES Flg Lk Inf Al
# CHECK:      .got              PROGBITS        00000000010021e0 0001e0 000020 00  WA  0   0  8
# CHECK-NEXT: .relro_padding    NOBITS          0000000001002200 000200 000e00 00  WA  0   0  1
# CHECK-NEXT: .data             PROGBITS        0000000001003200 000200 000006 00  WA  0   0  2

# CHECK-LABEL: Hex dump of section '.got':
# CHECK-NEXT:    0x010021e0 00000000 00000000 00000000 00000000
# CHECK-NEXT:    0x010021f0 00000000 00000000 00000000 01003205

# DISASM:      Disassembly of section .text:
# DISASM:      <_start>:
# DISASM-NEXT:   larl    %r1, 0x1003200
# DISASM-NEXT:   larl    %r1, 0x1003200
# DISASM-NEXT:   lgrl    %r1, 0x10021f8
# DISASM-NEXT:   lgrl    %r1, 0x10021f8

.data
.globl var_align
.hidden var_align
 .align 2
var_align:
 .long 0

.data
.globl var_unalign
.hidden var_unalign
 .align 2
 .byte 0
var_unalign:
 .byte 0

.text
.globl _start
.type _start, @function
_start:
 lgrl %r1, var_align@GOT
 lgrl %r1, var_align@GOT
 lgrl %r1, var_unalign@GOT
 lgrl %r1, var_unalign@GOT
