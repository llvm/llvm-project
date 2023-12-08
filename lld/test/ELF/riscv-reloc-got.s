# REQUIRES: riscv
# RUN: echo '.globl b; b:' | llvm-mc -filetype=obj -triple=riscv32 - -o %t1.o
# RUN: ld.lld -shared %t1.o -soname=t1.so -o %t1.so

# RUN: llvm-mc -filetype=obj -triple=riscv32 -position-independent %s -o %t.o
# RUN: ld.lld %t.o %t1.so -o %t
# RUN: llvm-readelf -S %t | FileCheck --check-prefix=SEC32 %s
# RUN: llvm-readobj -r %t | FileCheck --check-prefix=RELOC32 %s
# RUN: llvm-nm %t | FileCheck --check-prefix=NM32 %s
# RUN: llvm-readobj -x .got %t | FileCheck --check-prefix=HEX32 %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefix=DIS32 %s

# RUN: echo '.globl b; b:' | llvm-mc -filetype=obj -triple=riscv64 - -o %t1.o
# RUN: ld.lld -shared %t1.o -soname=t1.so -o %t1.so

# RUN: llvm-mc -filetype=obj -triple=riscv64 -position-independent %s -o %t.o
# RUN: ld.lld %t.o %t1.so -o %t
# RUN: llvm-readelf -S %t | FileCheck --check-prefix=SEC64 %s
# RUN: llvm-readobj -r %t | FileCheck --check-prefix=RELOC64 %s
# RUN: llvm-nm %t | FileCheck --check-prefix=NM64 %s
# RUN: llvm-readobj -x .got %t | FileCheck --check-prefix=HEX64 %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefix=DIS64 %s

# SEC32: .got PROGBITS         0001220c 00020c 00000c
# SEC64: .got PROGBITS 0000000000012358 000358 000018

# RELOC32:      .rela.dyn {
# RELOC32-NEXT:   0x12210 R_RISCV_32 b 0x0
# RELOC32-NEXT: }

# RELOC64:      .rela.dyn {
# RELOC64-NEXT:   0x12360 R_RISCV_64 b 0x0
# RELOC64-NEXT: }

# NM32: 00013218 d a
# NM64: 0000000000013370 d a

## .got[0] = _DYNAMIC
## .got[1] = 0 (relocated by R_RISCV_32/64 at runtime)
## .got[2] = a (filled at link time)
# HEX32: section '.got':
# HEX32: 0x0001220c ac210100 00000000 18320100 

# HEX64: section '.got':
# HEX64: 0x00012358 98220100 00000000 00000000 00000000
# HEX64: 0x00012368 70330100 00000000

## &.got[2]-. = 0x12214-0x1119c = 4096*1+120
# DIS32:      1119c: auipc a0, 1
# DIS32-NEXT:        lw a0, 120(a0)
## &.got[1]-. = 0x12210-0x111a4 = 4096*1+108
# DIS32:      111a4: auipc a0, 1
# DIS32-NEXT:        lw a0, 108(a0)

## &.got[2]-. = 0x12368-0x11288 = 4096*1+224
# DIS64:      11288: auipc a0, 1
# DIS64-NEXT:        ld a0, 224(a0)
## &.got[1]-. = 0x12360-0x11290 = 4096*1+208
# DIS64:      11290: auipc a0, 1
# DIS64-NEXT:        ld a0, 208(a0)

la a0,a
la a0,b

.data
a:
## An undefined reference of _GLOBAL_OFFSET_TABLE_ causes .got[0] to be
## allocated to store _DYNAMIC.
.long _GLOBAL_OFFSET_TABLE_ - .
