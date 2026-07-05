# REQUIRES: riscv
## The merged SHT_RISCV_ATTRIBUTES is placed at the first input
## SHT_RISCV_ATTRIBUTES.

# RUN: llvm-mc -filetype=obj -triple=riscv64 %s -o %t.o
# RUN: ld.lld -e 0 %t.o %t.o -o %t
# RUN: llvm-readelf -S %t | FileCheck %s

# CHECK:      Name              Type             Address          Off      Size   ES Flg Lk Inf Al
# CHECK:      .riscv.a          PROGBITS         0000000000000000 [[#%x,]] 000002 00      0   0  1
# CHECK-NEXT: .riscv.attributes RISCV_ATTRIBUTES 0000000000000000 [[#%x,]] 00001a 00      0   0  1
# CHECK-NEXT: .riscv.b          PROGBITS         0000000000000000 [[#%x,]] 000002 00      0   0  1

.section .riscv.a,""
.byte 0

.section .riscv.attributes,"",@0x70000003
.byte 0x41
.long .Lend-.riscv.attributes-1
.asciz "riscv"  # vendor
.Lbegin:
.byte 1  # Tag_File
.long .Lend-.Lbegin
.byte 5  # Tag_RISCV_arch
.asciz "rv64i2p0"
.Lend:

.section .riscv.b,""
.byte 0
