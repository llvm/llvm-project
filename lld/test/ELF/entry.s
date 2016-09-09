# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1
# RUN: ld.lld %t1 -o %t2
# RUN: llvm-readobj -file-headers %t2 | FileCheck -check-prefix=NOENTRY %s
# RUN: ld.lld %t1 -o %t2 -e entry
# RUN: llvm-readobj -file-headers %t2 | FileCheck -check-prefix=SYM %s
# RUN: ld.lld %t1 -shared -o %t2 -e entry
# RUN: llvm-readobj -file-headers %t2 | FileCheck -check-prefix=DSO %s
# RUN: ld.lld %t1 -o %t2 --entry=4096
# RUN: llvm-readobj -file-headers %t2 | FileCheck -check-prefix=DEC %s
# RUN: ld.lld %t1 -o %t2 --entry 0xcafe
# RUN: llvm-readobj -file-headers %t2 | FileCheck -check-prefix=HEX %s
# RUN: ld.lld %t1 -o %t2 -e 0777
# RUN: llvm-readobj -file-headers %t2 | FileCheck -check-prefix=OCT %s

# NOENTRY: Entry: 0x0
# SYM: Entry: 0x11000
# DSO: Entry: 0x1000
# DEC: Entry: 0x1000
# HEX: Entry: 0xCAFE
# OCT: Entry: 0x1FF

.globl entry
entry:
