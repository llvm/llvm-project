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
# RUN: llvm-objdump -d --mattr=+v8.5a --no-show-raw-insn %t.so | FileCheck --check-prefix BTIPACSO %s
# RUN: llvm-readelf --dynamic-table %t.so | FileCheck --check-prefix BTIPACDYN %s

# BTIPACSO: Disassembly of section .text:
# BTIPACSO: 0000000000010348 <func2>:
# BTIPACSO-NEXT:    10348:              bl      0x10380 <func3@plt>
# BTIPACSO-NEXT:                        ret
# BTIPACSO: 0000000000010350 <func3>:
# BTIPACSO-NEXT:    10350:              ret
# BTIPACSO: Disassembly of section .plt:
# BTIPACSO: 0000000000010360 <.plt>:
# BTIPACSO-NEXT:    10360:              bti     c
# BTIPACSO-NEXT:                        stp     x16, x30, [sp, #-16]!
# BTIPACSO-NEXT:                        adrp    x16, 0x30000
# BTIPACSO-NEXT:                        ldr     x17, [x16, #1136]
# BTIPACSO-NEXT:                        add     x16, x16, #1136
# BTIPACSO-NEXT:                        br      x17
# BTIPACSO-NEXT:                        nop
# BTIPACSO-NEXT:                        nop
# BTIPACSO: 0000000000010380 <func3@plt>:
# BTIPACSO-NEXT:    10380:              adrp    x16, 0x30000
# BTIPACSO-NEXT:                        ldr     x17, [x16, #1144]
# BTIPACSO-NEXT:                        add     x16, x16, #1144
# BTIPACSO-NEXT:                        br      x17

# BTIPACPROP:    Properties:    aarch64 feature: BTI, PAC

# BTIPACDYN:   0x0000000070000001 (AARCH64_BTI_PLT)
# BTIPACDYN-NOT:   0x0000000070000003 (AARCH64_PAC_PLT)

## Make an executable with both BTI and PAC properties. Expect:
## PLT[0] bti c as first instruction
## PLT[n] bti n as first instruction

# RUN: ld.lld %t.o %t3btipac.o %t.so -o %t.exe
# RUN: llvm-readelf -n %t.exe | FileCheck --check-prefix=BTIPACPROP %s
# RUN: llvm-objdump -d --mattr=+v8.5a --no-show-raw-insn %t.exe | FileCheck --check-prefix BTIPACEX %s
# RUN: llvm-readelf --dynamic-table %t.exe | FileCheck --check-prefix BTIPACDYNEX %s

# BTIPACEX: Disassembly of section .text:
# BTIPACEX: 0000000000210370 <func1>:
# BTIPACEX-NEXT:   210370:              bl      0x2103a0 <func2@plt>
# BTIPACEX-NEXT:                        ret
# BTIPACEX-NEXT:                        ret
# BTIPACEX: 000000000021037c <func3>:
# BTIPACEX-NEXT:   21037c:              ret
# BTIPACEX: Disassembly of section .plt:
# BTIPACEX: 0000000000210380 <.plt>:
# BTIPACEX-NEXT:   210380:              bti     c
# BTIPACEX-NEXT:                        stp     x16, x30, [sp, #-16]!
# BTIPACEX-NEXT:                        adrp    x16, 0x230000
# BTIPACEX-NEXT:                        ldr     x17, [x16, #1192]
# BTIPACEX-NEXT:                        add     x16, x16, #1192
# BTIPACEX-NEXT:                        br      x17
# BTIPACEX-NEXT:                        nop
# BTIPACEX-NEXT:                        nop
# BTIPACEX: 00000000002103a0 <func2@plt>:
# BTIPACEX-NEXT:   2103a0:              bti     c
# BTIPACEX-NEXT:                        adrp    x16, 0x230000
# BTIPACEX-NEXT:                        ldr     x17, [x16, #1200]
# BTIPACEX-NEXT:                        add     x16, x16, #1200
# BTIPACEX-NEXT:                        br      x17

# BTIPACDYNEX:   0x0000000070000001 (AARCH64_BTI_PLT)
# BTIPACDYNEX-NOT:   0x0000000070000003 (AARCH64_PAC_PLT)

## Check that combinations of BTI+PAC with 0 properties results in standard PLT

# RUN: ld.lld %t.o %t3.o %t.so -o %t.exe
# RUN: llvm-objdump -d --mattr=+v8.5a --no-show-raw-insn %t.exe | FileCheck --check-prefix EX %s
# RUN: llvm-readelf --dynamic-table %t.exe | FileCheck --check-prefix=NODYN %s

# EX: Disassembly of section .text:
# EX: 00000000002102e0 <func1>:
# EX-NEXT:   2102e0: bl      0x210310 <func2@plt>
# EX-NEXT:           ret
# EX-NEXT:           ret
# EX: 00000000002102ec <func3>:
# EX-NEXT:   2102ec: ret
# EX: Disassembly of section .plt:
# EX: 00000000002102f0 <.plt>:
# EX-NEXT:   2102f0: stp     x16, x30, [sp, #-16]!
# EX-NEXT:           adrp    x16, 0x230000
# EX-NEXT:           ldr     x17, [x16, #1024]
# EX-NEXT:           add     x16, x16, #1024
# EX-NEXT:           br      x17
# EX-NEXT:           nop
# EX-NEXT:           nop
# EX-NEXT:           nop
# EX: 0000000000210310 <func2@plt>:
# EX:        210310: adrp    x16, 0x230000
# EX-NEXT:           ldr     x17, [x16, #1032]
# EX-NEXT:           add     x16, x16, #1032
# EX-NEXT:           br      x17

# NODYN-NOT:   0x0000000070000001 (AARCH64_BTI_PLT)
# NODYN-NOT:   0x0000000070000003 (AARCH64_PAC_PLT)

## Check that combination of -z pac-plt and -z force-bti warns for the file that
## doesn't contain the BTI property, but generates PAC and BTI PLT sequences.
## The -z pac-plt doesn't warn as it is not required for correctness.
## Expect:
## PLT[0] bti c as first instruction
## PLT[n] bti n as first instruction, autia1716 before br x17

# RUN: ld.lld %t.o %t3.o %t.so -z pac-plt -z force-bti -o %t.exe 2>&1 | FileCheck --check-prefix=FORCE-WARN %s

# FORCE-WARN: aarch64-feature-btipac.s.tmp3.o: -z force-bti: file does not have GNU_PROPERTY_AARCH64_FEATURE_1_BTI property

# RUN: llvm-readelf -n %t.exe | FileCheck --check-prefix=BTIPACPROP %s
# RUN: llvm-objdump -d --mattr=+v8.5a --no-show-raw-insn %t.exe | FileCheck --check-prefix BTIPACEX2 %s
# RUN: llvm-readelf --dynamic-table %t.exe | FileCheck --check-prefix BTIPACDYN2 %s
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

# BTIPACEX2: Disassembly of section .text:
# BTIPACEX2: 0000000000210370 <func1>:
# BTIPACEX2-NEXT:   210370:              bl      0x2103a0 <func2@plt>
# BTIPACEX2-NEXT:                        ret
# BTIPACEX2-NEXT:                        ret
# BTIPACEX2: 000000000021037c <func3>:
# BTIPACEX2-NEXT:   21037c:              ret
# BTIPACEX2: Disassembly of section .plt:
# BTIPACEX2: 0000000000210380 <.plt>:
# BTIPACEX2-NEXT:   210380:              bti     c
# BTIPACEX2-NEXT:                        stp     x16, x30, [sp, #-16]!
# BTIPACEX2-NEXT:                        adrp    x16, 0x230000
# BTIPACEX2-NEXT:                        ldr     x17, [x16, #1208]
# BTIPACEX2-NEXT:                        add     x16, x16, #1208
# BTIPACEX2-NEXT:                        br      x17
# BTIPACEX2-NEXT:                        nop
# BTIPACEX2-NEXT:                        nop
# BTIPACEX2: 00000000002103a0 <func2@plt>:
# BTIPACEX2-NEXT:   2103a0:              bti     c
# BTIPACEX2-NEXT:                        adrp    x16, 0x230000
# BTIPACEX2-NEXT:                        ldr     x17, [x16, #1216]
# BTIPACEX2-NEXT:                        add     x16, x16, #1216
# BTIPACEX2-NEXT:                        autia1716
# BTIPACEX2-NEXT:                        br      x17

# BTIPACDYN2:        0x0000000070000001 (AARCH64_BTI_PLT)
# BTIPACDYN2-NEXT:   0x0000000070000003 (AARCH64_PAC_PLT)
