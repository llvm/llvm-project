# REQUIRES: sparc
# RUN: llvm-mc -filetype=obj -triple=sparcv9 %s -o %t.o
# RUN: ld.lld %t.o -o %t1
# RUN: llvm-readobj --file-headers %t1 | FileCheck --check-prefix=V9 %s
# RUN: ld.lld -m elf64_sparc %t.o -o %t2
# RUN: cmp %t1 %t2
# RUN: echo 'OUTPUT_FORMAT(elf64-sparc)' > %t.lds
# RUN: ld.lld -T %t.lds %t.o -o %t3
# RUN: llvm-readobj --file-headers %t3 | FileCheck --check-prefix=V9 %s

# V9:      ElfHeader {
# V9-NEXT:   Ident {
# V9-NEXT:     Magic: (7F 45 4C 46)
# V9-NEXT:     Class: 64-bit (0x2)
# V9-NEXT:     DataEncoding: BigEndian (0x2)
# V9-NEXT:     FileVersion: 1
# V9-NEXT:     OS/ABI: SystemV (0x0)
# V9-NEXT:     ABIVersion: 0
# V9-NEXT:     Unused: (00 00 00 00 00 00 00)
# V9-NEXT:   }
# V9-NEXT:   Type: Executable (0x2)
# V9-NEXT:   Machine: EM_SPARCV9 (0x2B)
# V9-NEXT:   Version: 1

.globl _start
_start:
