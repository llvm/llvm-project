# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-btipac1.s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-func3.s -o %t3.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %p/Inputs/aarch64-func3-btipac.s -o %t3btipac.o

## Build shared library with all inputs having BTI and PAC, expect PLT
## entries supporting both PAC and BTI. For a shared library this means:
## PLT[0] has bti c at start
## PLT[n] has autia1716 before br x17

# RUN: ld.lld %t1.o %t3btipac.o --shared --soname=t.so -o %t.so
# RUN: llvm-readelf -n %t.so | FileCheck --check-prefix BTIPACPROP %s
# RUN: llvm-objdump -d -mattr=+v8.5a --no-show-raw-insn %t.so | FileCheck --check-prefix BTIPACSO %s
# RUN: llvm-readelf --dynamic-table %t.so | FileCheck --check-prefix BTIPACDYN %s

# BTIPACSO: Disassembly of section .text:
# BTIPACSO: 0000000000010310 func2:
# BTIPACSO-NEXT:    10310: bl      #48 <func3@plt>
# BTIPACSO-NEXT:    10314: ret
# BTIPACSO: 0000000000010318 func3:
# BTIPACSO-NEXT:    10318: ret
# BTIPACSO: Disassembly of section .plt:
# BTIPACSO: 0000000000010320 .plt:
# BTIPACSO-NEXT:    10320: bti     c
# BTIPACSO-NEXT:    10324: stp     x16, x30, [sp, #-16]!
# BTIPACSO-NEXT:    10328: adrp    x16, #131072
# BTIPACSO-NEXT:    1032c: ldr     x17, [x16, #1096]
# BTIPACSO-NEXT:    10330: add     x16, x16, #1096
# BTIPACSO-NEXT:    10334: br      x17
# BTIPACSO-NEXT:    10338: nop
# BTIPACSO-NEXT:    1033c: nop
# BTIPACSO: 0000000000010340 func3@plt:
# BTIPACSO-NEXT:    10340: adrp    x16, #131072
# BTIPACSO-NEXT:    10344: ldr     x17, [x16, #1104]
# BTIPACSO-NEXT:    10348: add     x16, x16, #1104
# BTIPACSO-NEXT:    1034c: autia1716
# BTIPACSO-NEXT:    10350: br      x17
# BTIPACSO-NEXT:    10354: nop

# BTIPACPROP:    Properties:    aarch64 feature: BTI, PAC

# BTIPACDYN:   0x0000000070000001 (AARCH64_BTI_PLT)
# BTIPACDYN:   0x0000000070000003 (AARCH64_PAC_PLT)

## Make an executable with both BTI and PAC properties. Expect:
## PLT[0] bti c as first instruction
## PLT[n] bti n as first instruction, autia1716 before br x17

# RUN: ld.lld %t.o %t3btipac.o %t.so -o %t.exe
# RUN: llvm-readelf -n %t.exe | FileCheck --check-prefix=BTIPACPROP %s
# RUN: llvm-objdump -d -mattr=+v8.5a --no-show-raw-insn %t.exe | FileCheck --check-prefix BTIPACEX %s
# RUN: llvm-readelf --dynamic-table %t.exe | FileCheck --check-prefix BTIPACDYN %s

# BTIPACEX: Disassembly of section .text:
# BTIPACEX: 0000000000210338 func1:
# BTIPACEX-NEXT:   210338: bl      #56 <func2@plt>
# BTIPACEX-NEXT:   21033c: ret
# BTIPACEX-NEXT:   210340: ret
# BTIPACEX: 0000000000210344 func3:
# BTIPACEX-NEXT:   210344: ret
# BTIPACEX: Disassembly of section .plt:
# BTIPACEX: 0000000000210350 .plt:
# BTIPACEX-NEXT:   210350: bti     c
# BTIPACEX-NEXT:   210354: stp     x16, x30, [sp, #-16]!
# BTIPACEX-NEXT:   210358: adrp    x16, #131072
# BTIPACEX-NEXT:   21035c: ldr     x17, [x16, #1160]
# BTIPACEX-NEXT:   210360: add     x16, x16, #1160
# BTIPACEX-NEXT:   210364: br      x17
# BTIPACEX-NEXT:   210368: nop
# BTIPACEX-NEXT:   21036c: nop
# BTIPACEX: 0000000000210370 func2@plt:
# BTIPACEX-NEXT:   210370: bti     c
# BTIPACEX-NEXT:   210374: adrp    x16, #131072
# BTIPACEX-NEXT:   210378: ldr     x17, [x16, #1168]
# BTIPACEX-NEXT:   21037c: add     x16, x16, #1168
# BTIPACEX-NEXT:   210380: autia1716
# BTIPACEX-NEXT:   210384: br      x17

## Check that combinations of BTI+PAC with 0 properties results in standard PLT

# RUN: ld.lld %t.o %t3.o %t.so -o %t.exe
# RUN: llvm-objdump -d -mattr=+v8.5a --no-show-raw-insn %t.exe | FileCheck --check-prefix EX %s
# RUN: llvm-readelf --dynamic-table %t.exe | FileCheck --check-prefix=NODYN %s

# EX: Disassembly of section .text:
# EX: 00000000002102e0 func1:
# EX-NEXT:   2102e0: bl      #48 <func2@plt>
# EX-NEXT:   2102e4: ret
# EX-NEXT:   2102e8: ret
# EX: 00000000002102ec func3:
# EX-NEXT:   2102ec: ret
# EX: Disassembly of section .plt:
# EX: 00000000002102f0 .plt:
# EX-NEXT:   2102f0: stp     x16, x30, [sp, #-16]!
# EX-NEXT:   2102f4: adrp    x16, #131072
# EX-NEXT:   2102f8: ldr     x17, [x16, #1024]
# EX-NEXT:   2102fc: add     x16, x16, #1024
# EX-NEXT:   210300: br      x17
# EX-NEXT:   210304: nop
# EX-NEXT:   210308: nop
# EX-NEXT:   21030c: nop
# EX: 0000000000210310 func2@plt:
# EX:        210310: adrp    x16, #131072
# EX-NEXT:   210314: ldr     x17, [x16, #1032]
# EX-NEXT:   210318: add     x16, x16, #1032
# EX-NEXT:   21031c: br      x17

# NODYN-NOT:   0x0000000070000001 (AARCH64_BTI_PLT)
# NODYN-NOT:   0x0000000070000003 (AARCH64_PAC_PLT)

## Check that combination of --pac-plt and --force-bti warns for the file that
## doesn't contain the BTI property, but generates PAC and BTI PLT sequences.
## The --pac-plt doesn't warn as it is not required for correctness.

# RUN: ld.lld %t.o %t3.o %t.so --pac-plt --force-bti -o %t.exe 2>&1 | FileCheck --check-prefix=FORCE-WARN %s

# FORCE-WARN: aarch64-feature-btipac.s.tmp3.o: --force-bti: file does not have BTI property

# RUN: llvm-readelf -n %t.exe | FileCheck --check-prefix=BTIPACPROP %s
# RUN: llvm-objdump -d -mattr=+v8.5a --no-show-raw-insn %t.exe | FileCheck --check-prefix BTIPACEX %s
# RUN: llvm-readelf --dynamic-table %t.exe | FileCheck --check-prefix BTIPACDYN %s
.section ".note.gnu.property", "a"
.long 4
.long 0x10
.long 0x5
.asciz "GNU"

.long 0xc0000000 // GNU_PROPERTY_AARCH64_FEATURE_1_AND
.long 4
.long 3          // GNU_PROPERTY_AARCH64_FEATURE_1_BTI and PAC
.long 0

.text
.globl _start
.type func1,%function
func1:
  bl func2
  ret
.globl func3
.type func3,%function
  ret
