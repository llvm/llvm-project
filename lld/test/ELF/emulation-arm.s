# REQUIRES: arm
# RUN: llvm-mc -filetype=obj -triple=armv7-unknown-linux %s -o %tarm
# RUN: ld.lld -m armelf %tarm -o %t2arm
# RUN: llvm-readobj --file-headers %t2arm | FileCheck --check-prefix=ARM %s
# RUN: ld.lld -m armelf_linux_eabi %tarm -o %t3arm
# RUN: llvm-readobj --file-headers %t3arm | FileCheck --check-prefix=ARM %s
# RUN: ld.lld %tarm -o %t4arm
# RUN: llvm-readobj --file-headers %t4arm | FileCheck --check-prefix=ARM %s
# RUN: echo 'OUTPUT_FORMAT(elf32-littlearm)' > %t5arm.script
# RUN: ld.lld %t5arm.script %tarm -o %t5arm
# RUN: llvm-readobj --file-headers %t5arm | FileCheck --check-prefix=ARM %s
# ARM:      ElfHeader {
# ARM-NEXT:   Ident {
# ARM-NEXT:     Magic: (7F 45 4C 46)
# ARM-NEXT:     Class: 32-bit (0x1)
# ARM-NEXT:     DataEncoding: LittleEndian (0x1)
# ARM-NEXT:     FileVersion: 1
# ARM-NEXT:     OS/ABI: SystemV (0x0)
# ARM-NEXT:     ABIVersion: 0
# ARM-NEXT:     Unused: (00 00 00 00 00 00 00)
# ARM-NEXT:   }
# ARM-NEXT:   Type: Executable (0x2)
# ARM-NEXT:   Machine: EM_ARM (0x28)
# ARM-NEXT:   Version: 1

# RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv7aeb-none-linux-gnueabi %s -o %t.o
# RUN: ld.lld -marmelfb_linux_eabi %t.o -o %t1
# echo 'OUTPUT_FORMAT(elf32-bigarm)' > %t6arm.script
# ld.lld %tarm -o %t6arm
# RUN: llvm-readobj -h %t1 | FileCheck %s --check-prefix=ARMEB
# RUN: llvm-readobj -h %t1 | FileCheck %s --check-prefix=BE8

# RUN: ld.lld --be8 -marmelfb_linux_eabi %t.o -o %t1
# echo 'OUTPUT_FORMAT(elf32-bigarm)' > %t6arm.script
# ld.lld --be8 %tarm -o %t6arm
# RUN: llvm-readobj -h %t1 | FileCheck %s --check-prefix=ARMEB8

# ARMEB: ElfHeader {
# ARMEB-NEXT:   Ident {
# ARMEB-NEXT:     Magic: (7F 45 4C 46)
# ARMEB-NEXT:     Class: 32-bit (0x1)
# ARMEB-NEXT:     DataEncoding: BigEndian (0x2)
# ARMEB-NEXT:     FileVersion: 1
# ARMEB-NEXT:     OS/ABI: SystemV (0x0)
# ARMEB-NEXT:     ABIVersion: 0
# ARMEB-NEXT:     Unused: (00 00 00 00 00 00 00)
# ARMEB-NEXT:   }
# ARMEB-NEXT:   Type: Executable (0x2)
# ARMEB-NEXT:   Machine: EM_ARM (0x28)
# ARMEB-NEXT:   Version: 1

## Ensure that the EF_ARM_BE8 flag is not set for be32
## This will have to be modified on the be8 is implemented
# ARMEB:   Flags [ (0x5000200)
# ARMEB-NEXT:     0x200
# ARMEB-NEXT:     0x1000000
# ARMEB-NEXT:     0x4000000
# ARMEB-NEXT:   ]

# BE8-NOT: 0x800000

# ARMEB8: ElfHeader {
# ARMEB8-NEXT:   Ident {
# ARMEB8-NEXT:     Magic: (7F 45 4C 46)
# ARMEB8-NEXT:     Class: 32-bit (0x1)
# ARMEB8-NEXT:     DataEncoding: BigEndian (0x2)
# ARMEB8-NEXT:     FileVersion: 1
# ARMEB8-NEXT:     OS/ABI: SystemV (0x0)
# ARMEB8-NEXT:     ABIVersion: 0
# ARMEB8-NEXT:     Unused: (00 00 00 00 00 00 00)
# ARMEB8-NEXT:   }
# ARMEB8-NEXT:   Type: Executable (0x2)
# ARMEB8-NEXT:   Machine: EM_ARM (0x28)
# ARMEB8-NEXT:   Version: 1

## Ensure that the EF_ARM_BE8 flag is set for be8
# ARMEB8:   Flags [ (0x5800200)
# ARMEB8-NEXT:     0x200
# ARMEB8-NEXT:     0x800000
# ARMEB8-NEXT:     0x1000000
# ARMEB8-NEXT:     0x4000000
# ARMEB8-NEXT:   ]

.globl _start
_start:
