# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-pac1.s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-func3.s -o %t2.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-func3-pac.s -o %t3.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-func2.s -o %tno.o

## We do not add PAC support when the inputs don't have the .note.gnu.property
## field.

# RUN: ld.lld %tno.o %t3.o --shared -o %tno.so
# RUN: llvm-objdump -d --mattr=+v8.3a --no-show-raw-insn %tno.so | FileCheck --check-prefix=NOPAC %s
# RUN: llvm-readelf -x .got.plt %tno.so | FileCheck --check-prefix SOGOTPLT %s
# RUN: llvm-readelf --dynamic-table %tno.so | FileCheck --check-prefix NOPACDYN %s

# NOPAC: 00000000000102b8 <func2>:
# NOPAC-NEXT:    102b8: bl      0x102f0 <func3@plt>
# NOPAC-NEXT:           ret
# NOPAC: Disassembly of section .plt:
# NOPAC: 00000000000102d0 <.plt>:
# NOPAC-NEXT:    102d0: stp     x16, x30, [sp, #-16]!
# NOPAC-NEXT:           adrp    x16, 0x30000
# NOPAC-NEXT:           ldr     x17, [x16, #960]
# NOPAC-NEXT:           add     x16, x16, #960
# NOPAC-NEXT:           br      x17
# NOPAC-NEXT:           nop
# NOPAC-NEXT:           nop
# NOPAC-NEXT:           nop
# NOPAC: 00000000000102f0 <func3@plt>:
# NOPAC-NEXT:    102f0: adrp    x16, 0x30000
# NOPAC-NEXT:           ldr     x17, [x16, #968]
# NOPAC-NEXT:           add     x16, x16, #968
# NOPAC-NEXT:           br      x17

# SOGOTPLT: Hex dump of section '.got.plt':
# SOGOTPLT-NEXT: 0x000303b0 00000000 00000000 00000000 00000000
# SOGOTPLT-NEXT: 0x000303c0 00000000 00000000 d0020100 00000000

# NOPACDYN-NOT:   0x0000000070000001 (AARCH64_BTI_PLT)
# NOPACDYN-NOT:   0x0000000070000003 (AARCH64_PAC_PLT)

# RUN: ld.lld %t1.o %t3.o --shared --soname=t.so -o %t.so
# RUN: llvm-readelf -n %t.so | FileCheck --check-prefix PACPROP %s
# RUN: llvm-objdump -d --mattr=+v8.3a --no-show-raw-insn %t.so | FileCheck --check-prefix PACSO %s
# RUN: llvm-readelf -x .got.plt %t.so | FileCheck --check-prefix SOGOTPLT2 %s
# RUN: llvm-readelf --dynamic-table %t.so |  FileCheck --check-prefix PACDYN %s

## PAC has no effect on PLT[0], for PLT[N].
# PACSO: 0000000000010348 <func2>:
# PACSO-NEXT:    10348:         bl      0x10380 <func3@plt>
# PACSO-NEXT:                   ret
# PACSO: 0000000000010350 <func3>:
# PACSO-NEXT:    10350:         ret
# PACSO: Disassembly of section .plt:
# PACSO: 0000000000010360 <.plt>:
# PACSO-NEXT:    10360:         stp     x16, x30, [sp, #-16]!
# PACSO-NEXT:                   adrp    x16, 0x30000
# PACSO-NEXT:                   ldr     x17, [x16, #1120]
# PACSO-NEXT:                   add     x16, x16, #1120
# PACSO-NEXT:                   br      x17
# PACSO-NEXT:                   nop
# PACSO-NEXT:                   nop
# PACSO-NEXT:                   nop
# PACSO: 0000000000010380 <func3@plt>:
# PACSO-NEXT:    10380:         adrp    x16, 0x30000
# PACSO-NEXT:                   ldr     x17, [x16, #1128]
# PACSO-NEXT:                   add     x16, x16, #1128
# PACSO-NEXT:                   br      x17

# SOGOTPLT2: Hex dump of section '.got.plt':
# SOGOTPLT2-NEXT: 0x00030450 00000000 00000000 00000000 00000000
# SOGOTPLT2-NEXT: 0x00030460 00000000 00000000 60030100 00000000

# PACPROP: Properties:    aarch64 feature: PAC

# PACDYN-NOT:      0x0000000070000001 (AARCH64_BTI_PLT)
# PACDYN-NOT:      0x0000000070000003 (AARCH64_PAC_PLT)

## Turn on PAC entries with the -z pac-plt command line option. There are no
## warnings in this case as the choice to use PAC in PLT entries is orthogonal
## to the choice of using PAC in relocatable objects. The presence of the PAC
## .note.gnu.property is an indication of preference by the relocatable object.

# RUN: ld.lld %t.o %t2.o -z pac-plt %t.so -o %tpacplt.exe
# RUN: llvm-readelf -n %tpacplt.exe | FileCheck --check-prefix=PACPROP %s
# RUN: llvm-readelf --dynamic-table %tpacplt.exe | FileCheck --check-prefix PACDYN2 %s
# RUN: llvm-objdump -d --mattr=+v8.3a --no-show-raw-insn %tpacplt.exe | FileCheck --check-prefix PACPLT %s

# PACPLT: Disassembly of section .text:
# PACPLT: 0000000000210370 <func1>:
# PACPLT-NEXT:   210370:        bl      0x2103a0 <func2@plt>
# PACPLT-NEXT:                  ret
# PACPLT: 0000000000210378 <func3>:
# PACPLT-NEXT:   210378:        ret
# PACPLT: Disassembly of section .plt:
# PACPLT: 0000000000210380 <.plt>:
# PACPLT-NEXT:   210380:        stp     x16, x30, [sp, #-16]!
# PACPLT-NEXT:                  adrp    x16, 0x230000
# PACPLT-NEXT:                  ldr     x17, [x16, #1192]
# PACPLT-NEXT:                  add     x16, x16, #1192
# PACPLT-NEXT:                  br      x17
# PACPLT-NEXT:                  nop
# PACPLT-NEXT:                  nop
# PACPLT-NEXT:                  nop
# PACPLT: 00000000002103a0 <func2@plt>:
# PACPLT-NEXT:   2103a0:        adrp    x16, 0x230000
# PACPLT-NEXT:                  ldr     x17, [x16, #1200]
# PACPLT-NEXT:                  add     x16, x16, #1200
# PACPLT-NEXT:                  autia1716
# PACPLT-NEXT:                  br      x17
# PACPLT-NEXT:                  nop

# PACDYN2-NOT:      0x0000000070000001 (AARCH64_BTI_PLT)
# PACDYN2:      0x0000000070000003 (AARCH64_PAC_PLT)

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
