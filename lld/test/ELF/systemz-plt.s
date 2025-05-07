# REQUIRES: systemz
# RUN: echo '.globl bar, weak; .type bar,@function; .type weak,@function; bar: weak:' > %t1.s

# RUN: llvm-mc -filetype=obj -triple=s390x-unknown-linux %t1.s -o %t1.o
# RUN: ld.lld -shared %t1.o -soname=t1.so -o %t1.so
# RUN: llvm-mc -filetype=obj -triple=s390x-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o %t1.so -z separate-code -o %t
# RUN: llvm-readelf -S -s -r -x .got.plt %t | FileCheck %s
# RUN: llvm-objdump -d %t | FileCheck --check-prefixes=DIS %s

# CHECK: Section Headers:
# CHECK: .plt     PROGBITS 0000000001001020 001020 000060 00  AX  0   0 16
# CHECK: .got     PROGBITS 00000000010020d0 0020d0 000018 00  WA  0   0  8
# CHECK: .got.plt PROGBITS 00000000010030e8 0020e8 000010 00  WA  0   0  8

# CHECK: Relocation section '.rela.plt' at offset {{.*}} contains 2 entries:
# CHECK: 00000000010030e8 000000010000000b R_390_JMP_SLOT 0000000000000000 bar + 0
# CHECK: 00000000010030f0 000000020000000b R_390_JMP_SLOT 0000000000000000 weak + 0

## A canonical PLT has a non-zero st_value. bar and weak are called but their
## addresses are not taken, so a canonical PLT is not necessary.
# CHECK: Symbol table '.dynsym' contains 3 entries:
# CHECK-NEXT:   Num:    Value          Size Type    Bind   Vis       Ndx Name
# CHECK-NEXT:     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT   UND
# CHECK-NEXT:     1: 0000000000000000     0 FUNC    GLOBAL DEFAULT   UND bar
# CHECK-NEXT:     2: 0000000000000000     0 FUNC    WEAK   DEFAULT   UND weak

## The .got.plt slots relocated by .rela.plt point to .plt
## This is required by glibc.
# CHECK: Hex dump of section '.got.plt':
# CHECK-NEXT: 0x010030e8 00000000 0100104e 00000000 0100106e

# DIS: Disassembly of section .text:

# DIS: 0000000001001000 <_start>:
# DIS-NEXT: brasl	%r14, 0x1001012
# DIS-NEXT: brasl	%r14, 0x1001040
# DIS-NEXT: brasl	%r14, 0x1001060

# DIS: 0000000001001012 <foo>:
# DIS-NEXT: br	%r14

# DIS: Disassembly of section .plt:

# DIS: 0000000001001020 <.plt>:
# DIS-NEXT: 1001020: e3 10 f0 38 00 24    	stg	%r1, 56(%r15)
# DIS-NEXT: 1001026: c0 10 00 00 08 55          larl	%r1, 0x10020d0
# DIS-NEXT: 100102c: d2 07 f0 30 10 08    	mvc	48(8,%r15), 8(%r1)
# DIS-NEXT: 1001032: e3 10 10 10 00 04    	lg	%r1, 16(%r1)
# DIS-NEXT: 1001038: 07 f1        	br	%r1
# DIS-NEXT: 100103a: 07 00        	nopr
# DIS-NEXT: 100103c: 07 00        	nopr
# DIS-NEXT: 100103e: 07 00        	nopr
# DIS-NEXT: 1001040: c0 10 00 00 10 54    	larl	%r1, 0x10030e8
# DIS-NEXT: 1001046: e3 10 10 00 00 04    	lg	%r1, 0(%r1)
# DIS-NEXT: 100104c: 07 f1        	br	%r1
# DIS-NEXT: 100104e: 0d 10        	basr	%r1, 0
# DIS-NEXT: 1001050: e3 10 10 0c 00 14    	lgf	%r1, 12(%r1)
# DIS-NEXT: 1001056: c0 f4 ff ff ff e5    	jg	0x1001020
# DIS-NEXT: 100105c: 00 00        	<unknown>
# DIS-NEXT: 100105e: 00 00        	<unknown>
# DIS-NEXT: 1001060: c0 10 00 00 10 48    	larl	%r1, 0x10030f0
# DIS-NEXT: 1001066: e3 10 10 00 00 04    	lg	%r1, 0(%r1)
# DIS-NEXT: 100106c: 07 f1        	br	%r1
# DIS-NEXT: 100106e: 0d 10        	basr	%r1, 0
# DIS-NEXT: 1001070: e3 10 10 0c 00 14    	lgf	%r1, 12(%r1)
# DIS-NEXT: 1001076: c0 f4 ff ff ff d5    	jg	0x1001020
# DIS-NEXT: 100107c: 00 00        	<unknown>
# DIS-NEXT: 100107e: 00 18        	<unknown>

.global _start, foo, bar
.weak weak

_start:
  ## Use @plt to avoid generating direct references that would force
  ## allocation of a canonical PLT entry.
  brasl %r14, foo@plt
  brasl %r14, bar@plt
  brasl %r14, weak@plt

## foo is local and non-preemptable, no PLT is generated.
foo:
  br %r14
