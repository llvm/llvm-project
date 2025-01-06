# REQUIRES: aarch64

# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=aarch64 %p/Inputs/shared.s -o a.o
# RUN: ld.lld -shared a.o -o a.so

#--- ok.s
# RUN: llvm-mc -filetype=obj -triple=aarch64 ok.s -o ok.o

# RUN: ld.lld ok.o a.so -pie -o ok1
# RUN: llvm-readelf -r -S -x .got ok1 | FileCheck %s --check-prefix=OK1

# RUN: ld.lld ok.o a.o -pie -o ok2
# RUN: llvm-readelf -r -S -x .got -s ok2 | FileCheck %s --check-prefix=OK2

# OK1:      Offset            Info             Type                    Symbol's Value   Symbol's Name + Addend
# OK1-NEXT: 0000000000020380  0000000100000412 R_AARCH64_AUTH_GLOB_DAT 0000000000000000 bar + 0
# OK1-NEXT: 0000000000020388  0000000200000412 R_AARCH64_AUTH_GLOB_DAT 0000000000000000 zed + 0

## Symbol's values for bar and zed are equal since they contain no content (see Inputs/shared.s)
# OK2:         Offset            Info             Type                    Symbol's Value   Symbol's Name + Addend
# OK2-NEXT:    0000000000020320  0000000000000411 R_AARCH64_AUTH_RELATIVE 10260
# OK2-NEXT:    0000000000020328  0000000000000411 R_AARCH64_AUTH_RELATIVE 10260

# OK1:      Hex dump of section '.got':
# OK1-NEXT: 0x00020380 00000000 00000080 00000000 000000a0
##                                         ^^
##                                         0b10000000 bit 63 address diversity = true, bits 61..60 key = IA
##                                                           ^^
##                                                           0b10100000 bit 63 address diversity = true, bits 61..60 key = DA

# OK2: Symbol table '.symtab' contains {{.*}} entries:
# OK2:    Num:    Value          Size Type    Bind   Vis       Ndx Name
# OK2:         0000000000010260     0 FUNC    GLOBAL DEFAULT     6 bar
# OK2:         0000000000010260     0 NOTYPE  GLOBAL DEFAULT     6 zed

# OK2:         Hex dump of section '.got':
# OK2-NEXT:    0x00020320 00000000 00000080 00000000 000000a0
##                                         ^^
##                                         0b10000000 bit 63 address diversity = true, bits 61..60 key = IA
##                                                           ^^
##                                                           0b10100000 bit 63 address diversity = true, bits 61..60 key = DA

# RUN: llvm-objdump -d ok1 | FileCheck %s --check-prefix=OK1-ASM

# OK1-ASM:      <_start>:
# OK1-ASM-NEXT: adrp x0, 0x20000
# OK1-ASM-NEXT: ldr  x0, [x0, #0x380]
# OK1-ASM-NEXT: adrp x1, 0x20000
# OK1-ASM-NEXT: add  x1, x1, #0x380
# OK1-ASM-NEXT: adrp x0, 0x20000
# OK1-ASM-NEXT: ldr  x0, [x0, #0x388]
# OK1-ASM-NEXT: adrp x1, 0x20000
# OK1-ASM-NEXT: add  x1, x1, #0x388

# RUN: llvm-objdump -d ok2 | FileCheck %s --check-prefix=OK2-ASM

# OK2-ASM:         <_start>:
# OK2-ASM-NEXT:    adrp x0, 0x20000
# OK2-ASM-NEXT:    ldr  x0, [x0, #0x320]
# OK2-ASM-NEXT:    adrp x1, 0x20000
# OK2-ASM-NEXT:    add  x1, x1, #0x320
# OK2-ASM-NEXT:    adrp x0, 0x20000
# OK2-ASM-NEXT:    ldr  x0, [x0, #0x328]
# OK2-ASM-NEXT:    adrp x1, 0x20000
# OK2-ASM-NEXT:    add  x1, x1, #0x328

.globl _start
_start:
  adrp x0, :got_auth:bar
  ldr  x0, [x0, :got_auth_lo12:bar]
  adrp x1, :got_auth:bar
  add  x1, x1, :got_auth_lo12:bar
  adrp x0, :got_auth:zed
  ldr  x0, [x0, :got_auth_lo12:zed]
  adrp x1, :got_auth:zed
  add  x1, x1, :got_auth_lo12:zed

#--- ok-tiny.s
# RUN: llvm-mc -filetype=obj -triple=aarch64 ok-tiny.s -o ok-tiny.o

# RUN: ld.lld ok-tiny.o a.so -pie -o tiny1
# RUN: llvm-readelf -r -S -x .got tiny1 | FileCheck %s --check-prefix=TINY1

# RUN: ld.lld ok-tiny.o a.o -pie -o tiny2
# RUN: llvm-readelf -r -S -x .got -s tiny2 | FileCheck %s --check-prefix=TINY2

# TINY1:      Offset            Info             Type                    Symbol's Value   Symbol's Name + Addend
# TINY1-NEXT: 0000000000020368  0000000100000412 R_AARCH64_AUTH_GLOB_DAT 0000000000000000 bar + 0
# TINY1-NEXT: 0000000000020370  0000000200000412 R_AARCH64_AUTH_GLOB_DAT 0000000000000000 zed + 0

## Symbol's values for bar and zed are equal since they contain no content (see Inputs/shared.s)
# TINY2:         Offset            Info             Type                    Symbol's Value   Symbol's Name + Addend
# TINY2-NEXT:    0000000000020308  0000000000000411 R_AARCH64_AUTH_RELATIVE 10248
# TINY2-NEXT:    0000000000020310  0000000000000411 R_AARCH64_AUTH_RELATIVE 10248

# TINY1:      Hex dump of section '.got':
# TINY1-NEXT: 0x00020368 00000000 00000080 00000000 000000a0
##                                              ^^
##                                              0b10000000 bit 63 address diversity = true, bits 61..60 key = IA
##                                                                ^^
##                                                                0b10100000 bit 63 address diversity = true, bits 61..60 key = DA

# TINY2: Symbol table '.symtab' contains {{.*}} entries:
# TINY2:    Num:    Value          Size Type    Bind   Vis       Ndx Name
# TINY2:         0000000000010248     0 FUNC    GLOBAL DEFAULT     6 bar
# TINY2:         0000000000010248     0 NOTYPE  GLOBAL DEFAULT     6 zed

# TINY2:         Hex dump of section '.got':
# TINY2-NEXT:    0x00020308 00000000 00000080 00000000 000000a0
##                                              ^^
##                                              0b10000000 bit 63 address diversity = true, bits 61..60 key = IA
##                                                                ^^
##                                                                0b10100000 bit 63 address diversity = true, bits 61..60 key = DA

# RUN: llvm-objdump -d tiny1 | FileCheck %s --check-prefix=TINY1-ASM

# TINY1-ASM:      <_start>:
# TINY1-ASM-NEXT: adr x0, 0x20368
# TINY1-ASM-NEXT: ldr x1, 0x20370

# RUN: llvm-objdump -d tiny2 | FileCheck %s --check-prefix=TINY2-ASM

# TINY2-ASM:         <_start>:
# TINY2-ASM-NEXT:    adr x0, 0x20308
# TINY2-ASM-NEXT:    ldr x1, 0x20310

.globl _start
_start:
  adr  x0, :got_auth:bar
  ldr  x1, :got_auth:zed

#--- err.s
# RUN: llvm-mc -filetype=obj -triple=aarch64 err.s -o err.o
# RUN: not ld.lld err.o a.so -pie 2>&1 | FileCheck %s --check-prefix=ERR --implicit-check-not=error:
# ERR: error: both AUTH and non-AUTH GOT entries for 'bar' requested, but only one type of GOT entry per symbol is supported

.globl _start
_start:
  adrp x0, :got_auth:bar
  ldr  x0, [x0, :got_auth_lo12:bar]
  adrp x0, :got:bar
  ldr  x0, [x0, :got_lo12:bar]
