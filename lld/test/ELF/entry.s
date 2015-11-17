# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1
# RUN: not ld.lld2 %t1 -o %t2
# RUN: ld.lld2 %t1 -o %t2 -e _end

# RUN: ld.lld2 %t1 -o %t2 -e 4096
# RUN: llvm-readobj -file-headers %t2 | FileCheck -check-prefix=DEC %s
# RUN: ld.lld2 %t1 -o %t2 -e 0xcafe
# RUN: llvm-readobj -file-headers %t2 | FileCheck -check-prefix=HEX %s
# RUN: ld.lld2 %t1 -o %t2 -e 0777
# RUN: llvm-readobj -file-headers %t2 | FileCheck -check-prefix=OCT %s

# DEC: Entry: 0x1000
# HEX: Entry: 0xCAFE
# OCT: Entry: 0x1FF

.globl _end
_end:
