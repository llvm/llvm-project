// REQUIRES: x86

// RUN: llvm-mc -filetype=obj -triple=x86_64-windows %s -o %t-x86_64.obj
// RUN: llvm-mc -filetype=obj -triple=i686-windows %s -o %t-x86.obj
// RUN: lld-link -machine:amd64 -out:%t-x86_64.dll %t-x86_64.obj -dll -noentry
// RUN: lld-link -machine:x86 -out:%t-x86.dll %t-x86.obj -dll -noentry -safeseh:no

// RUN: llvm-readobj --hex-dump=.rdata %t-x86_64.dll | FileCheck %s -check-prefix=RDATA
// RUN: llvm-readobj --hex-dump=.rdata %t-x86.dll | FileCheck %s -check-prefix=RDATA
// RDATA: 78563412

// RUN: llvm-readobj --sections %t-x86_64.dll | FileCheck %s -check-prefix=SECTIONS
// RUN: llvm-readobj --sections %t-x86.dll | FileCheck %s -check-prefix=SECTIONS
// SECTIONS-NOT: .00cfg

        .section ".00cfg", "dr"
        .long 0x12345678
