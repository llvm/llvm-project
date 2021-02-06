# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readobj --file-headers %t | FileCheck --check-prefixes=AARCH64,LE %s
# RUN: ld.lld -m aarch64linux %t.o -o %t1
# RUN: llvm-readobj --file-headers %t1 | FileCheck --check-prefixes=AARCH64,LE %s
# RUN: ld.lld -m aarch64elf %t.o -o %t2
# RUN: llvm-readobj --file-headers %t2 | FileCheck --check-prefixes=AARCH64,LE %s
# RUN: echo 'OUTPUT_FORMAT(elf64-littleaarch64)' > %t.script
# RUN: ld.lld %t.script %t.o -o %t3
# RUN: llvm-readobj --file-headers %t3 | FileCheck --check-prefixes=AARCH64,LE %s
# RUN: ld.lld -m aarch64_elf64_le_vec %t.o -o %taosp
# RUN: llvm-readobj --file-headers %taosp | FileCheck --check-prefixes=AARCH64,LE %s

# AARCH64:      ElfHeader {
# AARCH64-NEXT:   Ident {
# AARCH64-NEXT:     Magic: (7F 45 4C 46)
# AARCH64-NEXT:     Class: 64-bit (0x2)
# LE-NEXT:          DataEncoding: LittleEndian (0x1)
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

# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-freebsd %s -o %tfbsd.o
# RUN: echo 'OUTPUT_FORMAT(elf64-aarch64-freebsd)' > %tfbsd.script
# RUN: ld.lld %tfbsd.script %tfbsd.o -o %tfbsd
# RUN: llvm-readobj --file-headers %tfbsd | FileCheck --check-prefix=AARCH64-FBSD %s
# AARCH64-FBSD:      ElfHeader {
# AARCH64-FBSD-NEXT:   Ident {
# AARCH64-FBSD-NEXT:     Magic: (7F 45 4C 46)
# AARCH64-FBSD-NEXT:     Class: 64-bit (0x2)
# AARCH64-FBSD-NEXT:     DataEncoding: LittleEndian (0x1)
# AARCH64-FBSD-NEXT:     FileVersion: 1
# AARCH64-FBSD-NEXT:     OS/ABI: FreeBSD (0x9)
# AARCH64-FBSD-NEXT:     ABIVersion: 0
# AARCH64-FBSD-NEXT:     Unused: (00 00 00 00 00 00 00)
# AARCH64-FBSD-NEXT:   }
# AARCH64-FBSD-NEXT:   Type: Executable (0x2)
# AARCH64-FBSD-NEXT:   Machine: EM_AARCH64 (0xB7)
# AARCH64-FBSD-NEXT:   Version: 1
# AARCH64-FBSD-NEXT:   Entry:
# AARCH64-FBSD-NEXT:   ProgramHeaderOffset: 0x40
# AARCH64-FBSD-NEXT:   SectionHeaderOffset:
# AARCH64-FBSD-NEXT:   Flags [ (0x0)
# AARCH64-FBSD-NEXT:   ]

.globl _start
_start:
