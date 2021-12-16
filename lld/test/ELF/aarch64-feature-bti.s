# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu --defsym CANONICAL_PLT=1 %s -o %tcanon.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-bti1.s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-func3.s -o %t2.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-func3-bti.s -o %t3.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-func2.s -o %tno.o

## We do not add BTI support when the inputs don't have the .note.gnu.property
## field.

# RUN: ld.lld %tno.o %t3.o --shared -o %tno.so
# RUN: llvm-objdump -d --mattr=+bti --no-show-raw-insn %tno.so | FileCheck --check-prefix=NOBTI %s
# RUN: llvm-readelf -x .got.plt %tno.so | FileCheck --check-prefix SOGOTPLT %s
# RUN: llvm-readelf --dynamic-table %tno.so | FileCheck --check-prefix NOBTIDYN %s

# NOBTIDYN-NOT:   0x0000000070000001 (AARCH64_BTI_PLT)
# NOBTIDYN-NOT:   0x0000000070000003 (AARCH64_PAC_PLT)

# NOBTI: 00000000000102b8 <func2>:
# NOBTI-NEXT:    102b8: bl      0x102f0 <func3@plt>
# NOBTI-NEXT:    102bc: ret
# NOBTI: Disassembly of section .plt:
# NOBTI: 00000000000102d0 <.plt>:
# NOBTI-NEXT:    102d0: stp     x16, x30, [sp, #-16]!
# NOBTI-NEXT:    102d4: adrp    x16, 0x30000
# NOBTI-NEXT:    102d8: ldr     x17, [x16, #960]
# NOBTI-NEXT:    102dc: add     x16, x16, #960
# NOBTI-NEXT:    102e0: br      x17
# NOBTI-NEXT:    102e4: nop
# NOBTI-NEXT:    102e8: nop
# NOBTI-NEXT:    102ec: nop
# NOBTI: 00000000000102f0 <func3@plt>:
# NOBTI-NEXT:    102f0: adrp    x16, 0x30000
# NOBTI-NEXT:    102f4: ldr     x17, [x16, #968]
# NOBTI-NEXT:    102f8: add     x16, x16, #968
# NOBTI-NEXT:    102fc: br      x17

## The .got.plt should be identical between the BTI and no BTI DSO PLT.
# SOGOTPLT: Hex dump of section '.got.plt'
# SOGOTPLT-NEXT:  0x000303b0 00000000 00000000 00000000 00000000
# SOGOTPLT-NEXT:  0x000303c0 00000000 00000000 d0020100 00000000

## Expect a bti c at the start of plt[0], the plt entries do not need bti c as
## their address doesn't escape the shared object, so they can't be indirectly
## called. Expect no other difference.

# RUN: ld.lld %t1.o %t3.o --shared --soname=t.so -o %t.so
# RUN: llvm-readelf -n %t.so | FileCheck --check-prefix BTIPROP %s
# RUN: llvm-objdump -d --mattr=+bti --no-show-raw-insn %t.so | FileCheck --check-prefix BTISO %s
# RUN: llvm-readelf -x .got.plt %t.so | FileCheck --check-prefix SOGOTPLT2 %s
# RUN: llvm-readelf --dynamic-table %t.so | FileCheck --check-prefix BTIDYN %s

# BTIPROP: Properties:    aarch64 feature: BTI

# BTIDYN:      0x0000000070000001 (AARCH64_BTI_PLT)
# BTIDYN-NOT:  0x0000000070000003 (AARCH64_PAC_PLT)

# BTISO: 0000000000010348 <func2>:
# BTISO-NEXT:    10348: bl      0x10380 <func3@plt>
# BTISO-NEXT:           ret
# BTISO: 0000000000010350 <func3>:
# BTISO-NEXT:    10350: ret
# BTISO: Disassembly of section .plt:
# BTISO: 0000000000010360 <.plt>:
# BTISO-NEXT:    10360: bti     c
# BTISO-NEXT:           stp     x16, x30, [sp, #-16]!
# BTISO-NEXT:           adrp    x16, 0x30000
# BTISO-NEXT:           ldr     x17, [x16, #1144]
# BTISO-NEXT:           add     x16, x16, #1144
# BTISO-NEXT:           br      x17
# BTISO-NEXT:           nop
# BTISO-NEXT:           nop
# BTISO: 0000000000010380 <func3@plt>:
# BTISO-NEXT:    10380: adrp    x16, 0x30000
# BTISO-NEXT:           ldr     x17, [x16, #1152]
# BTISO-NEXT:           add     x16, x16, #1152
# BTISO-NEXT:           br      x17

# SOGOTPLT2: Hex dump of section '.got.plt'
# SOGOTPLT2-NEXT:  0x00030468 00000000 00000000 00000000 00000000
# SOGOTPLT2-NEXT:  0x00030478 00000000 00000000 60030100 00000000

## Build an executable with all relocatable inputs having the BTI
## .note.gnu.property.

# RUN: ld.lld %t2.o --shared --soname=t2.so -o %t2.so

# RUN: ld.lld %t.o %t.so %t2.so -o %t.exe
# RUN: llvm-readelf --dynamic-table -n %t.exe | FileCheck --check-prefix=BTIPROP %s
# RUN: llvm-objdump -d --mattr=+bti --no-show-raw-insn %t.exe | FileCheck --check-prefix=EXECBTI %s

# EXECBTI: Disassembly of section .text:
# EXECBTI: 0000000000210348 <func1>:
# EXECBTI-NEXT:   210348: bl    0x210370 <func2@plt>
# EXECBTI-NEXT:           ret
# EXECBTI: Disassembly of section .plt:
# EXECBTI: 0000000000210350 <.plt>:
# EXECBTI-NEXT:   210350: bti   c
# EXECBTI-NEXT:           stp   x16, x30, [sp, #-16]!
# EXECBTI-NEXT:           adrp  x16, 0x230000
# EXECBTI-NEXT:           ldr   x17, [x16, #1160]
# EXECBTI-NEXT:           add   x16, x16, #1160
# EXECBTI-NEXT:           br    x17
# EXECBTI-NEXT:           nop
# EXECBTI-NEXT:           nop
# EXECBTI: 0000000000210370 <func2@plt>:
# EXECBTI-NEXT:   210370: adrp  x16, 0x230000
# EXECBTI-NEXT:           ldr   x17, [x16, #1168]
# EXECBTI-NEXT:           add   x16, x16, #1168
# EXECBTI-NEXT:           br    x17
# EXECBTI-NEXT:           nop
# EXECBTI-NEXT:           nop

## We expect a bti c in front of a canonical PLT entry because its address
## can escape the executable.
# RUN: ld.lld %tcanon.o %t.so %t2.so -o %t2.exe
# RUN: llvm-readelf --dynamic-table -n %t2.exe | FileCheck --check-prefix=BTIPROP %s
# RUN: llvm-objdump -d --mattr=+bti --no-show-raw-insn %t2.exe | FileCheck --check-prefix=EXECBTI2 %s
# EXECBTI2: 0000000000210380 <func2@plt>:
# EXECBTI2-NEXT:   210380: bti   c
# EXECBTI2-NEXT:           adrp  x16, 0x230000
# EXECBTI2-NEXT:           ldr   x17, [x16, #1184]
# EXECBTI2-NEXT:           add   x16, x16, #1184
# EXECBTI2-NEXT:           br    x17
# EXECBTI2-NEXT:           nop


## We expect the same for PIE, as the address of an ifunc can escape
# RUN: ld.lld --pie %t.o %t.so %t2.so -o %tpie.exe
# RUN: llvm-readelf -n %tpie.exe | FileCheck --check-prefix=BTIPROP %s
# RUN: llvm-readelf --dynamic-table -n %tpie.exe | FileCheck --check-prefix=BTIPROP %s
# RUN: llvm-objdump -d --mattr=+bti --no-show-raw-insn %tpie.exe | FileCheck --check-prefix=PIE %s

# PIE: Disassembly of section .text:
# PIE: 0000000000010348 <func1>:
# PIE-NEXT:    10348: bl     0x10370 <func2@plt>
# PIE-NEXT:           ret
# PIE: Disassembly of section .plt:
# PIE: 0000000000010350 <.plt>:
# PIE-NEXT:    10350: bti    c
# PIE-NEXT:           stp    x16, x30, [sp, #-16]!
# PIE-NEXT:           adrp   x16, 0x30000
# PIE-NEXT:           ldr    x17, [x16, #1176]
# PIE-NEXT:           add    x16, x16, #1176
# PIE-NEXT:           br     x17
# PIE-NEXT:           nop
# PIE-NEXT:           nop
# PIE: 0000000000010370 <func2@plt>:
# PIE-NEXT:    10370: adrp   x16, 0x30000
# PIE-NEXT:           ldr    x17, [x16, #1184]
# PIE-NEXT:           add    x16, x16, #1184
# PIE-NEXT:           br     x17
# PIE-NEXT:           nop
# PIE-NEXT:           nop

## Build and executable with not all relocatable inputs having the BTI
## .note.property, expect no bti c and no .note.gnu.property entry

# RUN: ld.lld %t.o %t2.o %t.so -o %tnobti.exe
# RUN: llvm-readelf --dynamic-table %tnobti.exe | FileCheck --check-prefix NOBTIDYN %s
# RUN: llvm-objdump -d --mattr=+bti --no-show-raw-insn %tnobti.exe | FileCheck --check-prefix=NOEX %s

# NOEX: Disassembly of section .text:
# NOEX: 00000000002102e0 <func1>:
# NOEX-NEXT:   2102e0: bl      0x210310 <func2@plt>
# NOEX-NEXT:           ret
# NOEX: 00000000002102e8 <func3>:
# NOEX-NEXT:   2102e8: ret
# NOEX: Disassembly of section .plt:
# NOEX: 00000000002102f0 <.plt>:
# NOEX-NEXT:   2102f0: stp     x16, x30, [sp, #-16]!
# NOEX-NEXT:           adrp    x16, 0x230000
# NOEX-NEXT:           ldr     x17, [x16, #1024]
# NOEX-NEXT:           add     x16, x16, #1024
# NOEX-NEXT:           br      x17
# NOEX-NEXT:           nop
# NOEX-NEXT:           nop
# NOEX-NEXT:           nop
# NOEX: 0000000000210310 <func2@plt>:
# NOEX-NEXT:   210310: adrp    x16, 0x230000
# NOEX-NEXT:           ldr     x17, [x16, #1032]
# NOEX-NEXT:           add     x16, x16, #1032
# NOEX-NEXT:           br      x17

## Force BTI entries with the -z force-bti command line option. Expect a warning
## from the file without the .note.gnu.property.

# RUN: ld.lld %t.o %t2.o -z force-bti %t.so -o %tforcebti.exe 2>&1 | FileCheck --check-prefix=FORCE-WARN %s
# RUN: not ld.lld %t.o %t2.o -z force-bti -z bti-report=error %t.so -o %tfailifnotbti.exe 2>&1 | FileCheck --check-prefix=BTI_REPORT-ERROR %s

# FORCE-WARN: aarch64-feature-bti.s.tmp2.o: -z force-bti: file does not have GNU_PROPERTY_AARCH64_FEATURE_1_BTI property
# BTI_REPORT-ERROR: aarch64-feature-bti.s.tmp2.o: -z bti-report: file does not have GNU_PROPERTY_AARCH64_FEATURE_1_BTI property
# BTI_REPORT-ERROR-EMPTY:

# RUN: llvm-readelf -n %tforcebti.exe | FileCheck --check-prefix=BTIPROP %s
# RUN: llvm-readelf --dynamic-table %tforcebti.exe | FileCheck --check-prefix BTIDYN %s
# RUN: llvm-objdump -d --mattr=+bti --no-show-raw-insn %tforcebti.exe | FileCheck --check-prefix=FORCE %s

# FORCE: Disassembly of section .text:
# FORCE: 0000000000210370 <func1>:
# FORCE-NEXT:   210370: bl      0x2103a0 <func2@plt>
# FORCE-NEXT:           ret
# FORCE: 0000000000210378 <func3>:
# FORCE-NEXT:   210378: ret
# FORCE: Disassembly of section .plt:
# FORCE: 0000000000210380 <.plt>:
# FORCE-NEXT:   210380: bti     c
# FORCE-NEXT:           stp     x16, x30, [sp, #-16]!
# FORCE-NEXT:           adrp    x16, 0x230000
# FORCE-NEXT:           ldr     x17, [x16, #1192]
# FORCE-NEXT:           add     x16, x16, #1192
# FORCE-NEXT:           br      x17
# FORCE-NEXT:           nop
# FORCE-NEXT:           nop
# FORCE: 00000000002103a0 <func2@plt>:
# FORCE-NEXT:   2103a0: adrp    x16, 0x230000
# FORCE-NEXT:           ldr     x17, [x16, #1200]
# FORCE-NEXT:           add     x16, x16, #1200
# FORCE-NEXT:           br      x17
# FORCE-NEXT:           nop
# FORCE-NEXT:           nop

.section ".note.gnu.property", "a"
.long 4
.long 0x10
.long 0x5
.asciz "GNU"

.long 0xc0000000 // GNU_PROPERTY_AARCH64_FEATURE_1_AND
.long 4
.long 1          // GNU_PROPERTY_AARCH64_FEATURE_1_BTI
.long 0

.text
.globl _start
.type func1,%function
func1:
.ifdef CANONICAL_PLT
  adrp x0, func2
  add  x0, x0, :lo12:func2
.else
  bl func2
.endif
  ret
