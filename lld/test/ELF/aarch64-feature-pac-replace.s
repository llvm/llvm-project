
### This file replace .note.gnu.property with aarch64 build attributes in order to confirm
### interoperability.
### (Still using gnu properties in the helper files)

# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-pac1.s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-pac1-replace.s -o %t1-ba.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-func3.s -o %t2.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-func3-pac.s -o %t3.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-func3-pac-replace.s -o %t3-ba.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-func2.s -o %tno.o

## We do not add PAC support when the inputs don't have the .note.gnu.property
## field.

# RUN: ld.lld %tno.o %t3.o --shared -o %tno.so
# RUN: ld.lld %tno.o %t3-ba.o --shared -o %tno-ba.so
# RUN: llvm-objdump --no-print-imm-hex -d --mattr=+v8.3a --no-show-raw-insn %tno.so | FileCheck --check-prefix=NOPAC %s
# RUN: llvm-objdump --no-print-imm-hex -d --mattr=+v8.3a --no-show-raw-insn %tno-ba.so | FileCheck --check-prefix=NOPAC %s
# RUN: llvm-readelf -x .got.plt %tno.so | FileCheck --check-prefix SOGOTPLT %s
# RUN: llvm-readelf -x .got.plt %tno-ba.so | FileCheck --check-prefix SOGOTPLT %s
# RUN: llvm-readelf --dynamic-table %tno.so | FileCheck --check-prefix NOPACDYN %s
# RUN: llvm-readelf --dynamic-table %tno-ba.so | FileCheck --check-prefix NOPACDYN %s

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
# RUN: ld.lld %t1-ba.o %t3-ba.o --shared --soname=t.so -o %t-ba.so
# RUN: llvm-readelf -n %t.so | FileCheck --check-prefix PACPROP %s
# RUN: llvm-readelf -n %t-ba.so | FileCheck --check-prefix PACPROP %s
# RUN: llvm-objdump --no-print-imm-hex -d --mattr=+v8.3a --no-show-raw-insn %t.so | FileCheck --check-prefix PACSO %s
# RUN: llvm-objdump --no-print-imm-hex -d --mattr=+v8.3a --no-show-raw-insn %t-ba.so | FileCheck --check-prefix PACSO %s
# RUN: llvm-readelf -x .got.plt %t.so | FileCheck --check-prefix SOGOTPLT2 %s
# RUN: llvm-readelf -x .got.plt %t-ba.so | FileCheck --check-prefix SOGOTPLT2 %s
# RUN: llvm-readelf --dynamic-table %t.so |  FileCheck --check-prefix PACDYN %s
# RUN: llvm-readelf --dynamic-table %t-ba.so |  FileCheck --check-prefix PACDYN %s

# PACPROP: Properties: aarch64 feature: PAC

# PACSO: Disassembly of section .text:
# PACSO: 0000000000010348 <func2>:
# PACSO-NEXT:   10348:      	bl	0x10380 <func3@plt>
# PACSO-NEXT:   1034c:      	ret
# PACSO: 0000000000010350 <func3>:
# PACSO-NEXT:   10350:      	ret
# PACSO: Disassembly of section .plt:
# PACSO: 0000000000010360 <.plt>:
# PACSO-NEXT:   10360:      	stp	x16, x30, [sp, #-16]!
# PACSO-NEXT:   10364:      	adrp	x16, 0x30000
# PACSO-NEXT:   10368:      	ldr	x17, [x16, #1120]
# PACSO-NEXT:   1036c:      	add	x16, x16, #1120
# PACSO-NEXT:   10370:      	br	x17
# PACSO-NEXT:   10374:      	nop
# PACSO-NEXT:   10378:      	nop
# PACSO-NEXT:   1037c:      	nop
# PACSO: 0000000000010380 <func3@plt>:
# PACSO-NEXT:   10380:      	adrp	x16, 0x30000
# PACSO-NEXT:   10384:      	ldr	x17, [x16, #1128]
# PACSO-NEXT:   10388:      	add	x16, x16, #1128
# PACSO-NEXT:   1038c:      	br	x17

# SOGOTPLT2: Hex dump of section '.got.plt':
# SOGOTPLT2-NEXT: 0x00030450 00000000 00000000 00000000 00000000
# SOGOTPLT2-NEXT: 0x00030460 00000000 00000000 60030100 00000000

# PACDYN-NOT:      0x0000000070000001 (AARCH64_BTI_PLT)
# PACDYN-NOT:      0x0000000070000003 (AARCH64_PAC_PLT)


# RUN: ld.lld %t.o %t2.o -z pac-plt %t.so -o %tpacplt.exe 2>&1 | FileCheck -DFILE=%t2.o --check-prefix WARN %s
# RUN: ld.lld %t.o %t2.o -z pac-plt %t-ba.so -o %tpacplt-ba.exe 2>&1 | FileCheck -DFILE=%t2.o --check-prefix WARN %s

# WARN: warning: [[FILE]]: -z pac-plt: file does not have GNU_PROPERTY_AARCH64_FEATURE_1_PAC property and no valid PAuth core info present for this link job


# RUN: llvm-readelf -n %tpacplt.exe | FileCheck --check-prefix=PACPROP %s
# RUN: llvm-readelf -n %tpacplt-ba.exe | FileCheck --check-prefix=PACPROP %s
# RUN: llvm-readelf --dynamic-table %tpacplt.exe | FileCheck --check-prefix PACDYN2 %s
# RUN: llvm-readelf --dynamic-table %tpacplt-ba.exe | FileCheck --check-prefix PACDYN2 %s
# RUN: llvm-objdump --no-print-imm-hex -d --mattr=+v8.3a --no-show-raw-insn %tpacplt.exe | FileCheck --check-prefix PACPLT %s
# RUN: llvm-objdump --no-print-imm-hex -d --mattr=+v8.3a --no-show-raw-insn %tpacplt-ba.exe | FileCheck --check-prefix PACPLT %s

# PACDYN2-NOT:      0x0000000070000001 (AARCH64_BTI_PLT)
# PACDYN2:      0x0000000070000003 (AARCH64_PAC_PLT)

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


.aeabi_subsection aeabi_feature_and_bits, optional, uleb128
.aeabi_attribute Tag_Feature_BTI, 0
.aeabi_attribute Tag_Feature_PAC, 1
.aeabi_attribute Tag_Feature_GCS, 0

.text
.globl _start
.type func1,%function
func1:
  bl func2
  ret
