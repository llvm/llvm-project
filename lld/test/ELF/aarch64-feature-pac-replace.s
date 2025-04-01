### This file replace .note.gnu.property with aarch64 build attributes in order to confirm
### interoperability.

# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-pac1.s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-func3.s -o %t2.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-func3-pac.s -o %t3.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-func2.s -o %tno.o

## We do not add PAC support when the inputs don't have the .note.gnu.property
## field.

# RUN: ld.lld %tno.o %t3.o --shared -o %tno.so
# RUN: llvm-objdump --no-print-imm-hex -d --mattr=+v8.3a --no-show-raw-insn %tno.so | FileCheck --check-prefix=NOPAC %s
# RUN: llvm-readelf -x .got.plt %tno.so | FileCheck --check-prefix SOGOTPLT %s
# RUN: llvm-readelf --dynamic-table %tno.so | FileCheck --check-prefix NOPACDYN %s

# NOPAC: Disassembly of section .text:
# NOPAC: 00000000000102b8 <func2>:
# NOPAC-NEXT:   102b8:      	bl	0x102f0 <func3@plt>
# NOPAC-NEXT:   102bc:      	ret
# NOPAC: 00000000000102c0 <func3>:
# NOPAC-NEXT:   102c0:      	ret
# NOPAC: Disassembly of section .plt:
# NOPAC: 00000000000102d0 <.plt>:
# NOPAC-NEXT:   102d0:      	stp	x16, x30, [sp, #-16]!
# NOPAC-NEXT:   102d4:      	adrp	x16, 0x30000
# NOPAC-NEXT:   102d8:      	ldr	x17, [x16, #960]
# NOPAC-NEXT:   102dc:      	add	x16, x16, #960
# NOPAC-NEXT:   102e0:      	br	x17
# NOPAC-NEXT:   102e4:      	nop
# NOPAC-NEXT:   102e8:      	nop
# NOPAC-NEXT:   102ec:      	nop
# NOPAC: 00000000000102f0 <func3@plt>:
# NOPAC-NEXT:   102f0:      	adrp	x16, 0x30000
# NOPAC-NEXT:   102f4:      	ldr	x17, [x16, #968]
# NOPAC-NEXT:   102f8:      	add	x16, x16, #968
# NOPAC-NEXT:   102fc:      	br	x17

# SOGOTPLT: Hex dump of section '.got.plt':
# SOGOTPLT-NEXT: 0x000303b0 00000000 00000000 00000000 00000000
# SOGOTPLT-NEXT: 0x000303c0 00000000 00000000 d0020100 00000000

# NOPACDYN-NOT:   0x0000000070000001 (AARCH64_BTI_PLT)
# NOPACDYN-NOT:   0x0000000070000003 (AARCH64_PAC_PLT)


# RUN: ld.lld %t1.o %t3.o --shared --soname=t.so -o %t.so
# RUN: llvm-readelf -n %t.so | FileCheck --check-prefix PACPROP %s
# RUN: llvm-objdump --no-print-imm-hex -d --mattr=+v8.3a --no-show-raw-insn %t.so | FileCheck --check-prefix PACSO %s
# RUN: llvm-readelf -x .got.plt %t.so | FileCheck --check-prefix SOGOTPLT2 %s
# RUN: llvm-readelf --dynamic-table %t.so |  FileCheck --check-prefix PACDYN %s

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

# WARN: warning: [[FILE]]: -z pac-plt: file does not have GNU_PROPERTY_AARCH64_FEATURE_1_PAC property and no valid PAuth core info present for this link job


# RUN: llvm-readelf -n %tpacplt.exe | FileCheck --check-prefix=PACPROP %s
# RUN: llvm-readelf --dynamic-table %tpacplt.exe | FileCheck --check-prefix PACDYN2 %s
# RUN: llvm-objdump --no-print-imm-hex -d --mattr=+v8.3a --no-show-raw-insn %tpacplt.exe | FileCheck --check-prefix PACPLT %s

# PACDYN2-NOT:      0x0000000070000001 (AARCH64_BTI_PLT)
# PACDYN2:      0x0000000070000003 (AARCH64_PAC_PLT)

# PACPLT: Disassembly of section .text:
# PACPLT: 0000000000210388 <func1>:
# PACPLT-NEXT:  210388:      	bl	0x2103c0 <func2@plt>
# PACPLT-NEXT:  21038c:      	ret
# PACPLT: 0000000000210390 <func3>:
# PACPLT-NEXT:  210390:      	ret
# PACPLT: Disassembly of section .plt:
# PACPLT: 00000000002103a0 <.plt>:
# PACPLT-NEXT:  2103a0:      	stp	x16, x30, [sp, #-16]!
# PACPLT-NEXT:  2103a4:      	adrp	x16, 0x230000 <func2+0x230000>
# PACPLT-NEXT:  2103a8:      	ldr	x17, [x16, #1224]
# PACPLT-NEXT:  2103ac:      	add	x16, x16, #1224
# PACPLT-NEXT:  2103b0:      	br	x17
# PACPLT-NEXT:  2103b4:      	nop
# PACPLT-NEXT:  2103b8:      	nop
# PACPLT-NEXT:  2103bc:      	nop
# PACPLT: 00000000002103c0 <func2@plt>:
# PACPLT-NEXT:  2103c0:      	adrp	x16, 0x230000 <func2+0x230000>
# PACPLT-NEXT:  2103c4:      	ldr	x17, [x16, #1232]
# PACPLT-NEXT:  2103c8:      	add	x16, x16, #1232
# PACPLT-NEXT:  2103cc:      	autia1716
# PACPLT-NEXT:  2103d0:      	br	x17
# PACPLT-NEXT:  2103d4:      	nop


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
