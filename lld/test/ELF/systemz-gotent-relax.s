# REQUIRES: systemz
## Test R_390_GOTENT optimization.

# RUN: llvm-mc -filetype=obj -triple=s390x-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t1 --no-apply-dynamic-relocs
# RUN: llvm-readelf -S -r -x .got.plt %t1 | FileCheck --check-prefixes=CHECK,NOAPPLY %s
# RUN: ld.lld %t.o -o %t1 --apply-dynamic-relocs
# RUN: llvm-readelf -S -r -x .got.plt %t1 | FileCheck --check-prefixes=CHECK,APPLY %s
# RUN: ld.lld %t.o -o %t1
# RUN: llvm-objdump --no-print-imm-hex -d %t1 | FileCheck --check-prefix=DISASM %s

## --no-relax disables GOT optimization.
# RUN: ld.lld --no-relax %t.o -o %t2
# RUN: llvm-objdump --no-print-imm-hex -d %t2 | FileCheck --check-prefix=NORELAX %s

## In our implementation, .got is retained even if all GOT-generating relocations are optimized.
# CHECK:      Name              Type            Address          Off    Size   ES Flg Lk Inf Al
# CHECK:      .iplt             PROGBITS        0000000001001240 000240 000020 00  AX  0   0 16
# CHECK-NEXT: .got              PROGBITS        0000000001002260 000260 000018 00  WA  0   0  8
# CHECK-NEXT: .relro_padding    NOBITS          0000000001002278 000278 000d88 00  WA  0   0  1
# CHECK-NEXT: .got.plt          PROGBITS        0000000001003278 000278 000008 00  WA  0   0  8

## There is one R_S390_IRELATIVE relocation.
# CHECK-LABEL: Relocation section '.rela.dyn' at offset {{.*}} contains 1 entries:
# CHECK:       0000000001003278  000000000000003d R_390_IRELATIVE                   10011e8

# CHECK-LABEL: Hex dump of section '.got.plt':
# NOAPPLY-NEXT:  0x01003278 00000000 00000000
# APPLY-NEXT:    0x01003278 00000000 010011e8

# DISASM:      Disassembly of section .text:
# DISASM: 00000000010011e0 <foo>:
# DISASM-NEXT:   nop
# DISASM: 00000000010011e4 <hid>:
# DISASM-NEXT:   nop
# DISASM: 00000000010011e8 <ifunc>:
# DISASM-NEXT:   br      %r14
# DISASM: 00000000010011ea <_start>:
# DISASM-NEXT:   larl    %r1, 0x10011e0
# DISASM-NEXT:   larl    %r1, 0x10011e0
# DISASM-NEXT:   larl    %r1, 0x10011e4
# DISASM-NEXT:   larl    %r1, 0x10011e4
# DISASM-NEXT:   lgrl    %r1, 0x1003278
# DISASM-NEXT:   lgrl    %r1, 0x1003278
# DISASM-NEXT:   larl    %r1, 0x10011e0
# DISASM-NEXT:   larl    %r1, 0x10011e0
# DISASM-NEXT:   larl    %r1, 0x10011e4
# DISASM-NEXT:   larl    %r1, 0x10011e4
# DISASM-NEXT:   lgrl    %r1, 0x1003278
# DISASM-NEXT:   lgrl    %r1, 0x1003278

# NORELAX-LABEL: <_start>:
# NORELAX-COUNT-12: lgrl

.text
.globl foo

.text
.globl foo
.type foo, @function
foo:
 nop

.globl hid
.hidden hid
.type hid, @function
hid:
 nop

.text
.type ifunc STT_GNU_IFUNC
.globl ifunc
.type ifunc, @function
ifunc:
 br %r14

.globl _start
.type _start, @function
_start:
 lgrl %r1, foo@GOT
 lgrl %r1, foo@GOT
 lgrl %r1, hid@GOT
 lgrl %r1, hid@GOT
 lgrl %r1, ifunc@GOT
 lgrl %r1, ifunc@GOT
 lgrl %r1, foo@GOT
 lgrl %r1, foo@GOT
 lgrl %r1, hid@GOT
 lgrl %r1, hid@GOT
 lgrl %r1, ifunc@GOT
 lgrl %r1, ifunc@GOT
