# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: ld.lld %t.o -r -o %t.ro
# RUN: llvm-readelf -S %t.ro | FileCheck %s
# RUN: llvm-objdump -s %t.ro | FileCheck %s --check-prefix=OBJDUMP

# RUN: ld.lld %t.o -o %t
# RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=CHECK-PDE

# CHECK:       [Nr] Name              Type            Address          Off    Size   ES Flg Lk Inf Al
# CHECK-NEXT:  [ 0]                   NULL            0000000000000000 000000 000000 00      0   0  0
# CHECK-NEXT:  [ 1] .text             PROGBITS        0000000000000000 000040 000000 00  AX  0   0  4
# CHECK-NEXT:  [ 2] .rodata.1         PROGBITS        0000000000000000 000040 000004 04  AM  0   0  4
# CHECK-NEXT:  [ 3] .rodata.2         PROGBITS        0000000000000000 000048 000008 08  AM  0   0  8
# CHECK-NEXT:  [ 4] .rodata.cst8      PROGBITS        0000000000000000 000050 000010 08  AM  0   0  1
# CHECK-NEXT:  [ 5] .rela.rodata.cst8 RELA            0000000000000000 000068 000030 18   I  9   4  8
# CHECK-NEXT:  [ 6] .cst4             PROGBITS        0000000000000000 000060 000008 04  AM  0   0  1
# CHECK-NEXT:  [ 7] .rela.cst4        RELA            0000000000000000 000098 000030 18   I  9   6  8

# OBJDUMP:      Contents of section .rodata.1:
# OBJDUMP-NEXT:  0000 42000000                             B...
# OBJDUMP-NEXT: Contents of section .rodata.2:
# OBJDUMP-NEXT:  0000 42000000 42000000                    B...B...
# OBJDUMP-NEXT: Contents of section .rodata.cst8:
# OBJDUMP-NEXT:  0000 00000000 00000000 00000000 00000000  ................
# OBJDUMP:      Contents of section .cst4:
# OBJDUMP-NEXT:  0000 00000000 00000000                    ........

# CHECK-PDE: [ 2] .cst4             PROGBITS        0000000000200140 000140 000008 04  AM  0   0  1

foo:

.section        .rodata.1,"aM",@progbits,4
.align  4
.long 0x42
.long 0x42
.long 0x42

.section        .rodata.2,"aM",@progbits,8
.align  8
.long   0x42
.long   0x42
.long   0x42
.long   0x42

## Test that we keep a SHT_REL[A] section which relocates a SHF_MERGE section
## in -r mode. The relocated SHF_MERGE section is handled as non-mergeable.
.section .rodata.cst8,"aM",@progbits,8,unique,0
.quad foo

.section .rodata.cst8,"aM",@progbits,8,unique,1
.quad foo

.section .cst4,"aM",@progbits,4,unique,0
.long foo
.section .cst4,"aM",@progbits,4,unique,1
.long foo
