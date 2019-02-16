# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-linux %s -o %taarch64
# RUN: ld.lld -m aarch64linux %taarch64 -o %t2aarch64
# RUN: llvm-readobj -file-headers %t2aarch64 | FileCheck --check-prefix=AARCH64 %s
# RUN: ld.lld -m aarch64elf %taarch64 -o %t3aarch64
# RUN: llvm-readobj -file-headers %t3aarch64 | FileCheck --check-prefix=AARCH64 %s
# RUN: ld.lld -m aarch64_elf64_le_vec %taarch64 -o %t4aarch64
# RUN: llvm-readobj -file-headers %t4aarch64 | FileCheck --check-prefix=AARCH64 %s
# RUN: ld.lld %taarch64 -o %t5aarch64
# RUN: llvm-readobj -file-headers %t5aarch64 | FileCheck --check-prefix=AARCH64 %s
# RUN: echo 'OUTPUT_FORMAT(elf64-littleaarch64)' > %t4aarch64.script
# RUN: ld.lld %t4aarch64.script %taarch64 -o %t4aarch64
# RUN: llvm-readobj -file-headers %t4aarch64 | FileCheck --check-prefix=AARCH64 %s
# AARCH64:      ElfHeader {
# AARCH64-NEXT:   Ident {
# AARCH64-NEXT:     Magic: (7F 45 4C 46)
# AARCH64-NEXT:     Class: 64-bit (0x2)
# AARCH64-NEXT:     DataEncoding: LittleEndian (0x1)
# AARCH64-NEXT:     FileVersion: 1
# AARCH64-NEXT:     OS/ABI: SystemV (0x0)
# AARCH64-NEXT:     ABIVersion: 0
# AARCH64-NEXT:     Unused: (00 00 00 00 00 00 00)
# AARCH64-NEXT:   }
# AARCH64-NEXT:   Type: Executable (0x2)
# AARCH64-NEXT:   Machine: EM_AARCH64 (0xB7)
# AARCH64-NEXT:   Version: 1
# AARCH64-NEXT:   Entry:
# AARCH64-NEXT:   ProgramHeaderOffset: 0x40
# AARCH64-NEXT:   SectionHeaderOffset:
# AARCH64-NEXT:   Flags [ (0x0)
# AARCH64-NEXT:   ]

.globl _start
_start:
