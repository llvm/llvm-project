# REQUIRES: ppc
# RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %s -o %tppc64
# RUN: ld.lld -m elf64ppc %tppc64 -o %t2ppc64
# RUN: llvm-readobj --file-headers %t2ppc64 | FileCheck --check-prefixes=CHECK,PPC64,LINUX,PPCBE %s
# RUN: ld.lld %tppc64 -o %t3ppc64
# RUN: llvm-readobj --file-headers %t3ppc64 | FileCheck --check-prefixes=CHECK,PPC64,LINUX,PPCBE %s
# RUN: echo 'OUTPUT_FORMAT(elf64-powerpc)' > %tppc64.script
# RUN: ld.lld %tppc64.script  %tppc64 -o %t4ppc64
# RUN: llvm-readobj --file-headers %t4ppc64 | FileCheck --check-prefixes=CHECK,PPC64,LINUX,PPCBE %s

# RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %s -o %tppc64le
# RUN: ld.lld -m elf64lppc %tppc64le -o %t2ppc64le
# RUN: llvm-readobj --file-headers %t2ppc64le | FileCheck --check-prefixes=CHECK,PPC64,LINUX,PPCLE %s
# RUN: ld.lld %tppc64le -o %t3ppc64le
# RUN: llvm-readobj --file-headers %t3ppc64le | FileCheck --check-prefixes=CHECK,PPC64,LINUX,PPCLE %s
# RUN: echo 'OUTPUT_FORMAT(elf64-powerpcle)' > %tppc64le.script
# RUN: ld.lld %tppc64le.script  %tppc64le -o %t4ppc64le
# RUN: llvm-readobj --file-headers %t4ppc64le | FileCheck --check-prefixes=CHECK,PPC64,LINUX,PPCLE %s

# RUN: llvm-mc -filetype=obj -triple=powerpc-unknown-linux %s -o %tppc32
# RUN: ld.lld -m elf32ppc %tppc32 -o %t2ppc32
# RUN: llvm-readobj --file-headers %t2ppc32 | FileCheck --check-prefixes=CHECK,PPC32,LINUX,PPCBE %s
# RUN: ld.lld %tppc32 -o %t3ppc32
# RUN: llvm-readobj --file-headers %t3ppc32 | FileCheck --check-prefixes=CHECK,PPC32,LINUX,PPCBE %s
# RUN: echo 'OUTPUT_FORMAT(elf32-powerpc)' > %tppc32.script
# RUN: ld.lld %tppc32.script  %tppc32 -o %t4ppc32
# RUN: llvm-readobj --file-headers %t4ppc32 | FileCheck --check-prefixes=CHECK,PPC32,LINUX,PPCBE %s
# RUN: ld.lld -m elf32ppclinux %tppc32 -o %t5ppc32
# RUN: llvm-readobj --file-headers %t5ppc32 | FileCheck --check-prefixes=CHECK,PPC32,LINUX,PPCBE %s

# RUN: llvm-mc -filetype=obj -triple=powerpcle-unknown-linux %s -o %tppc32le
# RUN: ld.lld -m elf32lppc %tppc32le -o %t2ppc32le
# RUN: llvm-readobj --file-headers %t2ppc32le | FileCheck --check-prefixes=CHECK,PPC32,LINUX,PPCLE %s
# RUN: ld.lld %tppc32le -o %t3ppc32le
# RUN: llvm-readobj --file-headers %t3ppc32le | FileCheck --check-prefixes=CHECK,PPC32,LINUX,PPCLE %s
# RUN: echo 'OUTPUT_FORMAT(elf32-powerpcle)' > %tppc32le.script
# RUN: ld.lld %tppc32le.script  %tppc32le -o %t4ppc32le
# RUN: llvm-readobj --file-headers %t4ppc32le | FileCheck --check-prefixes=CHECK,PPC32,LINUX,PPCLE %s
# RUN: ld.lld -m elf32lppclinux %tppc32le -o %t5ppc32le
# RUN: llvm-readobj --file-headers %t5ppc32le | FileCheck --check-prefixes=CHECK,PPC32,LINUX,PPCLE %s

# RUN: llvm-mc -filetype=obj -triple=powerpc-unknown-freebsd %s -o %tppc32fbsd
# RUN: echo 'OUTPUT_FORMAT(elf32-powerpc-freebsd)' > %tppc32fbsd.script
# RUN: ld.lld %tppc32fbsd.script  %tppc32fbsd -o %t2ppc32fbsd
# RUN: llvm-readobj --file-headers %t2ppc32fbsd | FileCheck --check-prefixes=CHECK,PPC32,FBSD,PPCBE %s

# RUN: llvm-mc -filetype=obj -triple=powerpcle-unknown-freebsd %s -o %tppc32fbsdle
# RUN: echo 'OUTPUT_FORMAT(elf32-powerpcle-freebsd)' > %tppc32fbsdle.script
# RUN: ld.lld %tppc32fbsdle.script  %tppc32fbsdle -o %t2ppc32fbsdle
# RUN: llvm-readobj --file-headers %t2ppc32fbsdle | FileCheck --check-prefixes=CHECK,PPC32,FBSD,PPCLE %s

# RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-freebsd %s -o %tppc64fbsd
# RUN: echo 'OUTPUT_FORMAT(elf64-powerpc-freebsd)' > %tppc64fbsd.script
# RUN: ld.lld %tppc64fbsd.script  %tppc64fbsd -o %t2ppc64fbsd
# RUN: llvm-readobj --file-headers %t2ppc64fbsd | FileCheck --check-prefixes=CHECK,PPC64,FBSD,PPCBE %s

# RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-freebsd %s -o %tppc64fbsdle
# RUN: echo 'OUTPUT_FORMAT(elf64-powerpcle-freebsd)' > %tppc64fbsdle.script
# RUN: ld.lld %tppc64fbsdle.script  %tppc64fbsdle -o %t2ppc64fbsdle
# RUN: llvm-readobj --file-headers %t2ppc64fbsdle | FileCheck --check-prefixes=CHECK,PPC64,FBSD,PPCLE %s

# CHECK:      ElfHeader {
# CHECK-NEXT:   Ident {
# CHECK-NEXT:     Magic: (7F 45 4C 46)

# PPC64-NEXT:     Class: 64-bit (0x2)
# PPC32-NEXT:     Class: 32-bit (0x1)

# PPCBE-NEXT:     DataEncoding: BigEndian (0x2)
# PPCLE-NEXT:     DataEncoding: LittleEndian (0x1)

# CHECK-NEXT:     FileVersion: 1

# LINUX-NEXT:     OS/ABI: SystemV (0x0)
# FBSD-NEXT:      OS/ABI: FreeBSD (0x9)

# CHECK-NEXT:     ABIVersion: 0
# CHECK-NEXT:     Unused: (00 00 00 00 00 00 00)
# CHECK-NEXT:   }
# CHECK-NEXT:   Type: Executable (0x2)

# PPC64-NEXT:   Machine: EM_PPC64 (0x15)
# PPC32-NEXT:   Machine: EM_PPC (0x14)

# CHECK-NEXT:   Version: 1
# CHECK-NEXT:   Entry:
# PPC64-NEXT:   ProgramHeaderOffset: 0x40
# PPC32-NEXT:   ProgramHeaderOffset: 0x34
# CHECK-NEXT:   SectionHeaderOffset:
# PPC64-NEXT:   Flags [ (0x2)
# PPC32-NEXT:   Flags [ (0x0)
# PPC64-NEXT:     0x2
# CHECK-NEXT:   ]
# PPC64-NEXT:   HeaderSize: 64
# PPC32-NEXT:   HeaderSize: 52
# PPC64-NEXT:   ProgramHeaderEntrySize: 56
# PPC32-NEXT:   ProgramHeaderEntrySize: 32
# CHECK-NEXT:   ProgramHeaderCount:
# PPC64-NEXT:   SectionHeaderEntrySize: 64
# PPC32-NEXT:   SectionHeaderEntrySize: 40
# CHECK-NEXT:   SectionHeaderCount:
# CHECK-NEXT:   StringTableSectionIndex:
# CHECK-NEXT: }

.globl _start
_start:
