# REQUIRES: x86
## .ctors/.dtors sections that have relocations are placed in
## .init_array/.fini_array when the link has .init_array/.fini_array input
## (--ctors-in-init-array, the default). Like GNU ld and mold, the entries of
## each moved section are reversed, and priority N becomes 65535-N.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 crtbegin.s -o crtbegin.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 ctors-only.s -o ctors-only.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 bad-size.s -o bad-size.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 bad-align.s -o bad-align.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 nobits.s -o nobits.o

# RUN: ld.lld a.o -o out
# RUN: llvm-readelf -S -x .init_array -x .fini_array out | FileCheck %s

# CHECK:      Name              Type            Address          Off    Size
# CHECK:      .init_array       INIT_ARRAY      {{.*}}           {{.*}} 000030
# CHECK-NEXT: .fini_array       FINI_ARRAY      {{.*}}           {{.*}} 000020
# CHECK-NOT:  .ctors
# CHECK-NOT:  .dtors

## .init_array is ordered: .init_array.5 (f1), .ctors.65525 (f2, priority 10),
## .init_array.100 (f3), then the unprioritized .init_array (f4) and .ctors
## (f6, f5 reversed to f5, f6) in input order.
# CHECK:      Hex dump of section '.init_array':
# CHECK-NEXT: 0x{{[0-9a-f]+}} 91112000 00000000 92112000 00000000
# CHECK-NEXT: 0x{{[0-9a-f]+}} 93112000 00000000 94112000 00000000
# CHECK-NEXT: 0x{{[0-9a-f]+}} 95112000 00000000 96112000 00000000
## .fini_array: .dtors.65525 (d1), .fini_array.100 (d2), .dtors reversed.
# CHECK:      Hex dump of section '.fini_array':
# CHECK-NEXT: 0x{{[0-9a-f]+}} 97112000 00000000 98112000 00000000
# CHECK-NEXT: 0x{{[0-9a-f]+}} 99112000 00000000 9a112000 00000000

## The last of --ctors-in-init-array and --no-ctors-in-init-array wins.
# RUN: ld.lld --no-ctors-in-init-array --ctors-in-init-array a.o -o out2
# RUN: cmp out out2

## --no-ctors-in-init-array keeps .ctors and .dtors separate.
# RUN: ld.lld --no-ctors-in-init-array a.o -o no
# RUN: llvm-readelf -S no | FileCheck %s --check-prefix=NO

# NO:     Name              Type
# NO-DAG: .init_array       INIT_ARRAY
# NO-DAG: .ctors            PROGBITS
# NO-DAG: .fini_array       FINI_ARRAY
# NO-DAG: .dtors            PROGBITS

## A legacy crtbegin.o has .ctors/.dtors with a -1 sentinel and no relocations.
## Those stay in .ctors/.dtors; the other .ctors/.dtors are still moved.
# RUN: ld.lld crtbegin.o a.o -o crt
# RUN: llvm-readelf -S -x .ctors -x .init_array crt | FileCheck %s --check-prefix=CRT

# CRT:      .ctors            PROGBITS        {{.*}}           {{.*}} 000008
# CRT-NEXT: .dtors            PROGBITS        {{.*}}           {{.*}} 000008
# CRT-NEXT: .init_array       INIT_ARRAY      {{.*}}           {{.*}} 000030
# CRT-NEXT: .fini_array       FINI_ARRAY      {{.*}}           {{.*}} 000020
# CRT:      Hex dump of section '.ctors':
# CRT-NEXT: 0x{{[0-9a-f]+}} ffffffff ffffffff
# CRT:      Hex dump of section '.init_array':
# CRT-NEXT: 0x{{[0-9a-f]+}} 91112000 00000000 92112000 00000000

## Without any .init_array/.fini_array input, .ctors/.dtors are left alone, as
## the runtime is expected to process them.
# RUN: ld.lld ctors-only.o -o ctors-only
# RUN: llvm-readelf -S -x .ctors ctors-only | FileCheck %s --check-prefix=ONLY

# ONLY-NOT:  .init_array
# ONLY:      .ctors            PROGBITS
# ONLY-NOT:  .init_array
# ONLY:      Hex dump of section '.ctors':
# ONLY-NEXT: 0x{{[0-9a-f]+}} 92112000 00000000 91112000 00000000

## A linker script that places .ctors in .init_array gets the same reversal.
# RUN: ld.lld -T script.lds a.o -o script
# RUN: llvm-readelf -S -x .init_array script | FileCheck %s --check-prefix=SCRIPT
# RUN: ld.lld -T script.lds --no-ctors-in-init-array a.o -o script-no
# RUN: llvm-readelf -x .init_array script-no | FileCheck %s --check-prefix=SCRIPT-NO

# SCRIPT:      .init_array       INIT_ARRAY
# SCRIPT:      Hex dump of section '.init_array':
# SCRIPT-NEXT: 0x{{[0-9a-f]+}} 01100000 00000000 02100000 00000000
# SCRIPT-NEXT: 0x{{[0-9a-f]+}} 03100000 00000000 04100000 00000000
# SCRIPT-NEXT: 0x{{[0-9a-f]+}} 05100000 00000000 06100000 00000000

## Without the option, the two entries of .ctors keep their input order.
# SCRIPT-NO:      Hex dump of section '.init_array':
# SCRIPT-NO-NEXT: 0x{{[0-9a-f]+}} 01100000 00000000 02100000 00000000
# SCRIPT-NO-NEXT: 0x{{[0-9a-f]+}} 03100000 00000000 04100000 00000000
# SCRIPT-NO-NEXT: 0x{{[0-9a-f]+}} 06100000 00000000 05100000 00000000

## -r leaves .ctors/.dtors alone.
# RUN: ld.lld -r a.o -o a.ro
# RUN: llvm-readelf -S a.ro | FileCheck %s --check-prefix=RELOCATABLE

# RELOCATABLE: .ctors            PROGBITS
# RELOCATABLE: .rela.ctors       RELA

## Moved sections must consist of whole entries with word-aligned relocations.
# RUN: not ld.lld bad-size.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=BAD-SIZE
# RUN: not ld.lld bad-align.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=BAD-ALIGN
# RUN: ld.lld --no-ctors-in-init-array bad-size.o -o /dev/null
# RUN: ld.lld --no-ctors-in-init-array bad-align.o -o /dev/null

# BAD-SIZE:  error: bad-size.o:(.ctors): section size (0x9) is not a multiple of 8; cannot place .ctors in .init_array
# BAD-ALIGN: error: bad-align.o:(.ctors+0x4): relocation is not aligned to a 8-byte entry; cannot place .ctors in .init_array

## SHT_NOBITS .ctors sections are ignored.
# RUN: ld.lld nobits.o -o nobits
# RUN: llvm-readelf -S nobits | FileCheck %s --check-prefix=NOBITS
# RUN: ld.lld -T script.lds nobits.o -o nobits-script

# NOBITS: .ctors NOBITS

#--- a.s
.globl _start, f1, f2, f3, f4, f5, f6, d1, d2, d3, d4
.text
_start:
  nop
f1:
  nop
f2:
  nop
f3:
  nop
f4:
  nop
f5:
  nop
f6:
  nop
d1:
  nop
d2:
  nop
d3:
  nop
d4:
  ret

.section .init_array.100, "aw", @init_array
  .quad f3

.section .ctors.65525, "aw", @progbits
  .quad f2

.section .init_array.5, "aw", @init_array
  .quad f1

.section .init_array, "aw", @init_array
  .quad f4

.section .ctors, "aw", @progbits
  .quad f6
  .quad f5

.section .fini_array.100, "aw", @fini_array
  .quad d2

.section .dtors.65525, "aw", @progbits
  .quad d1

.section .dtors, "aw", @progbits
  .quad d4
  .quad d3

#--- crtbegin.s
.section .ctors, "aw", @progbits
  .quad -1
.section .dtors, "aw", @progbits
  .quad -1

#--- ctors-only.s
.globl _start
_start:
  nop
c1:
  nop
c2:
  nop
.section .ctors, "aw", @progbits
  .quad c2
  .quad c1

#--- script.lds
SECTIONS {
  . = 0x1000;
  .text : { *(.text) }
  .init_array : {
    KEEP(*(SORT_BY_INIT_PRIORITY(.init_array.* .ctors.*)))
    KEEP(*(.init_array .ctors))
  }
  .fini_array : {
    KEEP(*(SORT_BY_INIT_PRIORITY(.fini_array.* .dtors.*)))
    KEEP(*(.fini_array .dtors))
  }
}

#--- bad-size.s
.globl _start
_start:
  nop
.section .init_array, "aw", @init_array
  .quad _start
.section .ctors, "aw", @progbits
  .quad _start
  .byte 0

#--- bad-align.s
.globl _start
_start:
  nop
.section .init_array, "aw", @init_array
  .quad _start
.section .ctors, "aw", @progbits
  .long 0
  .quad _start
  .long 0

#--- nobits.s
.globl _start
_start:
  nop
.section .init_array, "aw", @init_array
  .quad _start
.section .ctors, "aw", @nobits
  .space 16
