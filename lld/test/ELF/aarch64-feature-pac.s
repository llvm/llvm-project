# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-pac1.s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-func3.s -o %t2.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-func3-pac.s -o %t3.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-func2.s -o %tno.o

## We do not add PAC support when the inputs don't have the .note.gnu.property
## field.

# RUN: ld.lld %tno.o %t3.o --shared -o %tno.so
# RUN: llvm-objdump -d -mattr=+v8.3a --no-show-raw-insn %tno.so | FileCheck --check-prefix=NOPAC %s
# RUN: llvm-readelf -x .got.plt %tno.so | FileCheck --check-prefix SOGOTPLT %s
# RUN: llvm-readelf --dynamic-table %tno.so | FileCheck --check-prefix NOPACDYN %s

# NOPAC: 00000000000102b8 func2:
# NOPAC-NEXT:    102b8: bl      #56 <func3@plt>
# NOPAC-NEXT:    102bc: ret
# NOPAC: Disassembly of section .plt:
# NOPAC: 00000000000102d0 .plt:
# NOPAC-NEXT:    102d0: stp     x16, x30, [sp, #-16]!
# NOPAC-NEXT:    102d4: adrp    x16, #131072
# NOPAC-NEXT:    102d8: ldr     x17, [x16, #960]
# NOPAC-NEXT:    102dc: add     x16, x16, #960
# NOPAC-NEXT:    102e0: br      x17
# NOPAC-NEXT:    102e4: nop
# NOPAC-NEXT:    102e8: nop
# NOPAC-NEXT:    102ec: nop
# NOPAC: 00000000000102f0 func3@plt:
# NOPAC-NEXT:    102f0: adrp    x16, #131072
# NOPAC-NEXT:    102f4: ldr     x17, [x16, #968]
# NOPAC-NEXT:    102f8: add     x16, x16, #968
# NOPAC-NEXT:    102fc: br      x17

# NOPACDYN-NOT:   0x0000000070000001 (AARCH64_BTI_PLT)
# NOPACDYN-NOT:   0x0000000070000003 (AARCH64_PAC_PLT)

# RUN: ld.lld %t1.o %t3.o --shared --soname=t.so -o %t.so
# RUN: llvm-readelf -n %t.so | FileCheck --check-prefix PACPROP %s
# RUN: llvm-objdump -d -mattr=+v8.3a --no-show-raw-insn %t.so | FileCheck --check-prefix PACSO %s
# RUN: llvm-readelf -x .got.plt %t.so | FileCheck --check-prefix SOGOTPLT2 %s
# RUN: llvm-readelf --dynamic-table %t.so |  FileCheck --check-prefix PACDYN %s

## PAC has no effect on PLT[0], for PLT[N] autia1716 is used to authenticate
## the address in x17 (context in x16) before branching to it. The dynamic
## loader is responsible for calling pacia1716 on the entry.
# PACSO: 0000000000010310 func2:
# PACSO-NEXT:    10310: bl      #48 <func3@plt>
# PACSO-NEXT:    10314: ret
# PACSO: Disassembly of section .plt:
# PACSO: 0000000000010320 .plt:
# PACSO-NEXT:    10320: stp     x16, x30, [sp, #-16]!
# PACSO-NEXT:    10324: adrp    x16, #131072
# PACSO-NEXT:    10328: ldr     x17, [x16, #1080]
# PACSO-NEXT:    1032c: add     x16, x16, #1080
# PACSO-NEXT:    10330: br      x17
# PACSO-NEXT:    10334: nop
# PACSO-NEXT:    10338: nop
# PACSO-NEXT:    1033c: nop
# PACSO: 0000000000010340 func3@plt:
# PACSO-NEXT:    10340: adrp    x16, #131072
# PACSO-NEXT:    10344: ldr     x17, [x16, #1088]
# PACSO-NEXT:    10348: add     x16, x16, #1088
# PACSO-NEXT:    1034c: autia1716
# PACSO-NEXT:    10350: br      x17
# PACSO-NEXT:    10354: nop

# SOGOTPLT: Hex dump of section '.got.plt':
# SOGOTPLT-NEXT: 0x000303b0 00000000 00000000 00000000 00000000
# SOGOTPLT-NEXT: 0x000303c0 00000000 00000000 d0020100 00000000

# SOGOTPLT2: Hex dump of section '.got.plt':
# SOGOTPLT2-NEXT: 0x00030428 00000000 00000000 00000000 00000000
# SOGOTPLT2-NEXT: 0x00030438 00000000 00000000 20030100 00000000

# PACPROP: Properties:    aarch64 feature: PAC

# PACDYN-NOT:      0x0000000070000001 (AARCH64_BTI_PLT)
# PACDYN:          0x0000000070000003 (AARCH64_PAC_PLT)

## Turn on PAC entries with the --pac-plt command line option. There are no
## warnings in this case as the choice to use PAC in PLT entries is orthogonal
## to the choice of using PAC in relocatable objects. The presence of the PAC
## .note.gnu.property is an indication of preference by the relocatable object.

# RUN: ld.lld %t.o %t2.o --pac-plt %t.so -o %tpacplt.exe
# RUN: llvm-readelf -n %tpacplt.exe | FileCheck --check-prefix=PACPROP %s
# RUN: llvm-readelf --dynamic-table %tpacplt.exe | FileCheck --check-prefix PACDYN %s
# RUN: llvm-objdump -d -mattr=+v8.3a --no-show-raw-insn %tpacplt.exe | FileCheck --check-prefix PACPLT %s

# PACPLT: Disassembly of section .text:
# PACPLT: 0000000000210338 func1:
# PACPLT-NEXT:   210338: bl      #56 <func2@plt>
# PACPLT-NEXT:   21033c: ret
# PACPLT: 0000000000210340 func3:
# PACPLT-NEXT:   210340: ret
# PACPLT: Disassembly of section .plt:
# PACPLT: 0000000000210350 .plt:
# PACPLT-NEXT:   210350: stp     x16, x30, [sp, #-16]!
# PACPLT-NEXT:   210354: adrp    x16, #131072
# PACPLT-NEXT:   210358: ldr     x17, [x16, #1144]
# PACPLT-NEXT:   21035c: add     x16, x16, #1144
# PACPLT-NEXT:   210360: br      x17
# PACPLT-NEXT:   210364: nop
# PACPLT-NEXT:   210368: nop
# PACPLT-NEXT:   21036c: nop
# PACPLT: 0000000000210370 func2@plt:
# PACPLT-NEXT:   210370: adrp    x16, #131072
# PACPLT-NEXT:   210374: ldr     x17, [x16, #1152]
# PACPLT-NEXT:   210378: add     x16, x16, #1152
# PACPLT-NEXT:   21037c: autia1716
# PACPLT-NEXT:   210380: br      x17
# PACPLT-NEXT:   210384: nop


.section ".note.gnu.property", "a"
.long 4
.long 0x10
.long 0x5
.asciz "GNU"

.long 0xc0000000 // GNU_PROPERTY_AARCH64_FEATURE_1_AND
.long 4
.long 2          // GNU_PROPERTY_AARCH64_FEATURE_1_PAC
.long 0

.text
.globl _start
.type func1,%function
func1:
  bl func2
  ret
