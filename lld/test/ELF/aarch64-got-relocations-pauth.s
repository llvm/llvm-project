# REQUIRES: aarch64

# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=aarch64 %p/Inputs/shared.s -o a.o
# RUN: ld.lld -shared a.o -o a.so

#--- ok.s

# RUN: llvm-mc -filetype=obj -triple=aarch64 ok.s -o ok.o

# RUN: ld.lld ok.o a.so -pie -o external
# RUN: llvm-readelf -r -S -x .got external | FileCheck %s --check-prefix=EXTERNAL

# RUN: ld.lld ok.o a.o -pie -o local
# RUN: llvm-readelf -r -S -x .got -s local | FileCheck %s --check-prefix=LOCAL

# EXTERNAL:      Offset            Info             Type                    Symbol's Value   Symbol's Name + Addend
# EXTERNAL-NEXT: 0000000000020380  000000010000e201 R_AARCH64_AUTH_GLOB_DAT 0000000000000000 bar + 0
# EXTERNAL-NEXT: 0000000000020388  000000020000e201 R_AARCH64_AUTH_GLOB_DAT 0000000000000000 zed + 0

## Symbol's values for bar and zed are equal since they contain no content (see Inputs/shared.s)
# LOCAL:         Offset            Info             Type                    Symbol's Value   Symbol's Name + Addend
# LOCAL-NEXT:    0000000000020320  0000000000000411 R_AARCH64_AUTH_RELATIVE 10260
# LOCAL-NEXT:    0000000000020328  0000000000000411 R_AARCH64_AUTH_RELATIVE 10260

# EXTERNAL:      Hex dump of section '.got':
# EXTERNAL-NEXT: 0x00020380 00000000 00000080 00000000 000000a0
##                                         ^^
##                                         0b10000000 bit 63 address diversity = true, bits 61..60 key = IA
##                                                           ^^
##                                                           0b10100000 bit 63 address diversity = true, bits 61..60 key = DA

# LOCAL: Symbol table '.symtab' contains {{.*}} entries:
# LOCAL:    Num:    Value          Size Type    Bind   Vis       Ndx Name
# LOCAL:         0000000000010260     0 FUNC    GLOBAL DEFAULT     6 bar
# LOCAL:         0000000000010260     0 NOTYPE  GLOBAL DEFAULT     6 zed

# LOCAL:         Hex dump of section '.got':
# LOCAL-NEXT:    0x00020320 00000000 00000080 00000000 000000a0
##                                         ^^
##                                         0b10000000 bit 63 address diversity = true, bits 61..60 key = IA
##                                                           ^^
##                                                           0b10100000 bit 63 address diversity = true, bits 61..60 key = DA

# RUN: llvm-objdump -d external | FileCheck %s --check-prefix=EXTERNAL-ASM

# EXTERNAL-ASM:      <_start>:
# EXTERNAL-ASM-NEXT: adrp x0, 0x20000
# EXTERNAL-ASM-NEXT: ldr  x0, [x0, #0x380]
# EXTERNAL-ASM-NEXT: adrp x1, 0x20000
# EXTERNAL-ASM-NEXT: add  x1, x1, #0x380
# EXTERNAL-ASM-NEXT: adrp x0, 0x20000
# EXTERNAL-ASM-NEXT: ldr  x0, [x0, #0x388]
# EXTERNAL-ASM-NEXT: adrp x1, 0x20000
# EXTERNAL-ASM-NEXT: add  x1, x1, #0x388

# RUN: llvm-objdump -d local | FileCheck %s --check-prefix=LOCAL-ASM

# LOCAL-ASM:         <_start>:
# LOCAL-ASM-NEXT:    adrp x0, 0x20000
# LOCAL-ASM-NEXT:    ldr  x0, [x0, #0x320]
# LOCAL-ASM-NEXT:    adrp x1, 0x20000
# LOCAL-ASM-NEXT:    add  x1, x1, #0x320
# LOCAL-ASM-NEXT:    adrp x0, 0x20000
# LOCAL-ASM-NEXT:    ldr  x0, [x0, #0x328]
# LOCAL-ASM-NEXT:    adrp x1, 0x20000
# LOCAL-ASM-NEXT:    add  x1, x1, #0x328

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

#--- err.s

# RUN: llvm-mc -filetype=obj -triple=aarch64 err.s -o err.o

# RUN: not ld.lld err.o a.so -pie -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

# ERR:      error: both AUTH and non-AUTH GOT entries for 'bar' requested, but only one type of GOT entry per symbol is supported
# ERR-NEXT: >>> defined in a.so
# ERR-NEXT: >>> referenced by err.o:(.text+0x8)
# ERR:      error: both AUTH and non-AUTH GOT entries for 'bar' requested, but only one type of GOT entry per symbol is supported
# ERR-NEXT: >>> defined in a.so
# ERR-NEXT: >>> referenced by err.o:(.text+0xc)

.globl _start
_start:
  adrp x0, :got_auth:bar
  ldr  x0, [x0, :got_auth_lo12:bar]
  adrp x0, :got:bar
  ldr  x0, [x0, :got_lo12:bar]
