# REQUIRES: riscv

# RUN: llvm-mc -filetype=obj -triple=riscv32 %s -o %t.32.o
## rv32 IE
# RUN: ld.lld -shared %t.32.o -o %t.32.so
# RUN: llvm-readobj -r -d %t.32.so | FileCheck --check-prefix=IE32-REL %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.32.so | FileCheck --check-prefixes=IE,IE32 %s
## rv32 IE -> LE
# RUN: ld.lld %t.32.o -o %t.32
# RUN: llvm-readelf -r %t.32 | FileCheck --check-prefix=NOREL %s
# RUN: llvm-readelf -x .got %t.32 | FileCheck --check-prefix=LE32-GOT %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.32 | FileCheck --check-prefixes=LE,LE32 %s

# RUN: llvm-mc -filetype=obj -triple=riscv64 %s -o %t.64.o
## rv64 IE
# RUN: ld.lld -shared %t.64.o -o %t.64.so
# RUN: llvm-readobj -r -d %t.64.so | FileCheck --check-prefix=IE64-REL %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.64.so | FileCheck --check-prefixes=IE,IE64 %s
## rv64 IE -> LE
# RUN: ld.lld %t.64.o -o %t.64
# RUN: llvm-readelf -r %t.64 | FileCheck --check-prefix=NOREL %s
# RUN: llvm-readelf -x .got %t.64 | FileCheck --check-prefix=LE64-GOT %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.64 | FileCheck --check-prefixes=LE,LE64 %s

# IE32-REL:      FLAGS STATIC_TLS
# IE32-REL:      .rela.dyn {
# IE32-REL-NEXT:   0x221C R_RISCV_TLS_TPREL32 - 0xC
# IE32-REL-NEXT:   0x2218 R_RISCV_TLS_TPREL32 a 0x0
# IE32-REL-NEXT: }

# IE64-REL:      FLAGS STATIC_TLS
# IE64-REL:      .rela.dyn {
# IE64-REL-NEXT:   0x2378 R_RISCV_TLS_TPREL64 - 0xC
# IE64-REL-NEXT:   0x2370 R_RISCV_TLS_TPREL64 a 0x0
# IE64-REL-NEXT: }

## rv32: &.got[0] - . = 0x2218 - . = 4096*1+116
## rv64: &.got[0] - . = 0x2378 - . = 4096*1+208
# IE:              auipc a4, 1
# IE32-NEXT:       lw a4, 116(a4)
# IE64-NEXT:       ld a4, 208(a4)
# IE-NEXT:         add a4, a4, tp
## rv32: &.got[1] - . = 0x221c - . = 4096*1+108
## rv64: &.got[1] - . = 0x2378 - . = 4096*1+204
# IE:              auipc a5, 1
# IE32-NEXT:       lw a5, 108(a5)
# IE64-NEXT:       ld a5, 204(a5)
# IE-NEXT:         add a5, a5, tp

# NOREL: no relocations

# a@tprel = st_value(a) = 0x8
# b@tprel = st_value(a) = 0xc
# LE32-GOT: section '.got':
# LE32-GOT-NEXT: 0x0001212c 00000000 08000000 0c000000
# LE64-GOT: section '.got':
# LE64-GOT-NEXT: 0x000121e0 00000000 00000000 08000000 00000000
# LE64-GOT-NEXT: 0x000121f0 0c000000 00000000

## rv32: &.got[0] - . = 0x12130 - 0x11114 = 4096*1+28
## rv64: &.got[0] - . = 0x121e8 - 0x111c8 = 4096*1+32
# LE32:      11114: auipc a4, 1
# LE32-NEXT:        lw a4, 28(a4)
# LE64:      111c8: auipc a4, 1
# LE64-NEXT:        ld a4, 32(a4)
# LE-NEXT:          add a4, a4, tp
## rv32: &.got[1] - . = 0x12134 - 0x11120 = 4096*1+20
## rv64: &.got[1] - . = 0x121f0 - 0x111d4 = 4096*1+28
# LE32:      11120: auipc a5, 1
# LE32-NEXT:        lw a5, 20(a5)
# LE64:      111d4: auipc a5, 1
# LE64-NEXT:        ld a5, 28(a5)
# LE-NEXT:          add a5, a5, tp

la.tls.ie a4,a
add a4,a4,tp
la.tls.ie a5,b
add a5,a5,tp

.section .tbss
.globl a
.zero 8
a:
.zero 4
b:
