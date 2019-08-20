# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-bti1.s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-func3.s -o %t2.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-func3-bti.s -o %t3.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-func2.s -o %tno.o

## We do not add BTI support when the inputs don't have the .note.gnu.property
## field.

# RUN: ld.lld %tno.o %t3.o --shared -o %tno.so
# RUN: llvm-objdump -d -mattr=+bti --no-show-raw-insn %tno.so | FileCheck --check-prefix=NOBTI %s
# RUN: llvm-readelf -x .got.plt %tno.so | FileCheck --check-prefix SOGOTPLT %s
# RUN: llvm-readelf --dynamic-table %tno.so | FileCheck --check-prefix NOBTIDYN %s

# NOBTIDYN-NOT:   0x0000000070000001 (AARCH64_BTI_PLT)
# NOBTIDYN-NOT:   0x0000000070000003 (AARCH64_PAC_PLT)

# NOBTI: 00000000000102b8 func2:
# NOBTI-NEXT:    102b8: bl      #56 <func3@plt>
# NOBTI-NEXT:    102bc: ret
# NOBTI: Disassembly of section .plt:
# NOBTI: 00000000000102d0 .plt:
# NOBTI-NEXT:    102d0: stp     x16, x30, [sp, #-16]!
# NOBTI-NEXT:    102d4: adrp    x16, #131072
# NOBTI-NEXT:    102d8: ldr     x17, [x16, #960]
# NOBTI-NEXT:    102dc: add     x16, x16, #960
# NOBTI-NEXT:    102e0: br      x17
# NOBTI-NEXT:    102e4: nop
# NOBTI-NEXT:    102e8: nop
# NOBTI-NEXT:    102ec: nop
# NOBTI: 00000000000102f0 func3@plt:
# NOBTI-NEXT:    102f0: adrp    x16, #131072
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
# RUN: llvm-objdump -d -mattr=+bti --no-show-raw-insn %t.so | FileCheck --check-prefix BTISO %s
# RUN: llvm-readelf -x .got.plt %t.so | FileCheck --check-prefix SOGOTPLT2 %s
# RUN: llvm-readelf --dynamic-table %t.so | FileCheck --check-prefix BTIDYN %s

# BTIPROP: Properties:    aarch64 feature: BTI

# BTIDYN:      0x0000000070000001 (AARCH64_BTI_PLT)
# BTIDYN-NOT:  0x0000000070000003 (AARCH64_PAC_PLT)

# BTISO: 0000000000010310 func2:
# BTISO-NEXT:    10310: bl      #48 <func3@plt>
# BTISO-NEXT:    10314: ret
# BTISO: Disassembly of section .plt:
# BTISO: 0000000000010320 .plt:
# BTISO-NEXT:    10320: bti     c
# BTISO-NEXT:    10324: stp     x16, x30, [sp, #-16]!
# BTISO-NEXT:    10328: adrp    x16, #131072
# BTISO-NEXT:    1032c: ldr     x17, [x16, #1072]
# BTISO-NEXT:    10330: add     x16, x16, #1072
# BTISO-NEXT:    10334: br      x17
# BTISO-NEXT:    10338: nop
# BTISO-NEXT:    1033c: nop
# BTISO: 0000000000010340 func3@plt:
# BTISO-NEXT:    10340: adrp    x16, #131072
# BTISO-NEXT:    10344: ldr     x17, [x16, #1080]
# BTISO-NEXT:    10348: add     x16, x16, #1080
# BTISO-NEXT:    1034c: br      x17

# SOGOTPLT2: Hex dump of section '.got.plt'
# SOGOTPLT2-NEXT:  0x00030420 00000000 00000000 00000000 00000000
# SOGOTPLT2-NEXT:  0x00030430 00000000 00000000 20030100 00000000

## Build an executable with all relocatable inputs having the BTI
## .note.gnu.property. We expect a bti c in front of all PLT entries as the
## address of a PLT entry can escape an executable.

# RUN: ld.lld %t2.o --shared --soname=t2.so -o %t2.so

# RUN: ld.lld %t.o %t.so %t2.so -o %t.exe
# RUN: llvm-readelf --dynamic-table -n %t.exe | FileCheck --check-prefix=BTIPROP %s
# RUN: llvm-objdump -d -mattr=+bti --no-show-raw-insn %t.exe | FileCheck --check-prefix=EXECBTI %s

# EXECBTI: Disassembly of section .text:
# EXECBTI: 0000000000210310 func1:
# EXECBTI-NEXT:   210310: bl      #48 <func2@plt>
# EXECBTI-NEXT:   210314: ret
# EXECBTI: Disassembly of section .plt:
# EXECBTI: 0000000000210320 .plt:
# EXECBTI-NEXT:   210320: bti     c
# EXECBTI-NEXT:   210324: stp     x16, x30, [sp, #-16]!
# EXECBTI-NEXT:   210328: adrp    x16, #131072
# EXECBTI-NEXT:   21032c: ldr     x17, [x16, #1112]
# EXECBTI-NEXT:   210330: add     x16, x16, #1112
# EXECBTI-NEXT:   210334: br      x17
# EXECBTI-NEXT:   210338: nop
# EXECBTI-NEXT:   21033c: nop
# EXECBTI: 0000000000210340 func2@plt:
# EXECBTI-NEXT:   210340: bti     c
# EXECBTI-NEXT:   210344: adrp    x16, #131072
# EXECBTI-NEXT:   210348: ldr     x17, [x16, #1120]
# EXECBTI-NEXT:   21034c: add     x16, x16, #1120
# EXECBTI-NEXT:   210350: br      x17
# EXECBTI-NEXT:   210354: nop

## We expect the same for PIE, as the address of an ifunc can escape
# RUN: ld.lld --pie %t.o %t.so %t2.so -o %tpie.exe
# RUN: llvm-readelf -n %tpie.exe | FileCheck --check-prefix=BTIPROP %s
# RUN: llvm-readelf --dynamic-table -n %tpie.exe | FileCheck --check-prefix=BTIPROP %s
# RUN: llvm-objdump -d -mattr=+bti --no-show-raw-insn %tpie.exe | FileCheck --check-prefix=PIE %s

# PIE: Disassembly of section .text:
# PIE: 0000000000010310 func1:
# PIE-NEXT:    10310: bl      #48 <func2@plt>
# PIE-NEXT:    10314: ret
# PIE: Disassembly of section .plt:
# PIE: 0000000000010320 .plt:
# PIE-NEXT:    10320: bti     c
# PIE-NEXT:    10324: stp     x16, x30, [sp, #-16]!
# PIE-NEXT:    10328: adrp    x16, #131072
# PIE-NEXT:    1032c: ldr     x17, [x16, #1112]
# PIE-NEXT:    10330: add     x16, x16, #1112
# PIE-NEXT:    10334: br      x17
# PIE-NEXT:    10338: nop
# PIE-NEXT:    1033c: nop
# PIE: 0000000000010340 func2@plt:
# PIE-NEXT:    10340: bti     c
# PIE-NEXT:    10344: adrp    x16, #131072
# PIE-NEXT:    10348: ldr     x17, [x16, #1120]
# PIE-NEXT:    1034c: add     x16, x16, #1120
# PIE-NEXT:    10350: br      x17
# PIE-NEXT:    10354: nop

## Build and executable with not all relocatable inputs having the BTI
## .note.property, expect no bti c and no .note.gnu.property entry

# RUN: ld.lld %t.o %t2.o %t.so -o %tnobti.exe
# RUN: llvm-readelf --dynamic-table %tnobti.exe | FileCheck --check-prefix NOBTIDYN %s
# RUN: llvm-objdump -d -mattr=+bti --no-show-raw-insn %tnobti.exe | FileCheck --check-prefix=NOEX %s

# NOEX: Disassembly of section .text:
# NOEX: 00000000002102e0 func1:
# NOEX-NEXT:   2102e0: bl      #48 <func2@plt>
# NOEX-NEXT:   2102e4: ret
# NOEX: 00000000002102e8 func3:
# NOEX-NEXT:   2102e8: ret
# NOEX: Disassembly of section .plt:
# NOEX: 00000000002102f0 .plt:
# NOEX-NEXT:   2102f0: stp     x16, x30, [sp, #-16]!
# NOEX-NEXT:   2102f4: adrp    x16, #131072
# NOEX-NEXT:   2102f8: ldr     x17, [x16, #1024]
# NOEX-NEXT:   2102fc: add     x16, x16, #1024
# NOEX-NEXT:   210300: br      x17
# NOEX-NEXT:   210304: nop
# NOEX-NEXT:   210308: nop
# NOEX-NEXT:   21030c: nop
# NOEX: 0000000000210310 func2@plt:
# NOEX-NEXT:   210310: adrp    x16, #131072
# NOEX-NEXT:   210314: ldr     x17, [x16, #1032]
# NOEX-NEXT:   210318: add     x16, x16, #1032
# NOEX-NEXT:   21031c: br      x17

## Force BTI entries with the --force-bti command line option. Expect a warning
## from the file without the .note.gnu.property.

# RUN: ld.lld %t.o %t2.o --force-bti %t.so -o %tforcebti.exe 2>&1 | FileCheck --check-prefix=FORCE-WARN %s

# FORCE-WARN: aarch64-feature-bti.s.tmp2.o: --force-bti: file does not have BTI property


# RUN: llvm-readelf -n %tforcebti.exe | FileCheck --check-prefix=BTIPROP %s
# RUN: llvm-readelf --dynamic-table %tforcebti.exe | FileCheck --check-prefix BTIDYN %s
# RUN: llvm-objdump -d -mattr=+bti --no-show-raw-insn %tforcebti.exe | FileCheck --check-prefix=FORCE %s

# FORCE: Disassembly of section .text:
# FORCE: 0000000000210338 func1:
# FORCE-NEXT:   210338: bl      #56 <func2@plt>
# FORCE-NEXT:   21033c: ret
# FORCE: 0000000000210340 func3:
# FORCE-NEXT:   210340: ret
# FORCE: Disassembly of section .plt:
# FORCE: 0000000000210350 .plt:
# FORCE-NEXT:   210350: bti     c
# FORCE-NEXT:   210354: stp     x16, x30, [sp, #-16]!
# FORCE-NEXT:   210358: adrp    x16, #131072
# FORCE-NEXT:   21035c: ldr     x17, [x16, #1144]
# FORCE-NEXT:   210360: add     x16, x16, #1144
# FORCE-NEXT:   210364: br      x17
# FORCE-NEXT:   210368: nop
# FORCE-NEXT:   21036c: nop
# FORCE: 0000000000210370 func2@plt:
# FORCE-NEXT:   210370: bti     c
# FORCE-NEXT:   210374: adrp    x16, #131072
# FORCE-NEXT:   210378: ldr     x17, [x16, #1152]
# FORCE-NEXT:   21037c: add     x16, x16, #1152
# FORCE-NEXT:   210380: br      x17
# FORCE-NEXT:   210384: nop

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
  bl func2
  ret
