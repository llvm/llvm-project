# REQUIRES: aarch64
# RUN: llvm-mc -triple=aarch64-windows -filetype=obj -o %t-arm64.obj %s
# RUN: llvm-mc -triple=arm64ec-windows -filetype=obj -o %t-arm64ec.obj %s

# RUN: lld-link -machine:arm64x -dll -noentry %t-arm64.obj %t-arm64ec.obj -debug -build-id -Brepro -out:%t.dll
# RUN: llvm-readobj --hex-dump=.test %t.dll | FileCheck %s
# CHECK: 0x180003000 3c100000 3c100000

# RUN: lld-link -machine:arm64ec -dll -noentry %t-arm64.obj %t-arm64ec.obj -debug -build-id -Brepro -out:%t-ec.dll
# RUN: llvm-readobj --hex-dump=.test %t-ec.dll | FileCheck %s

.section .test,"dr"
.rva __buildid

.section .bss,"bw",discard,__buildid
.global __buildid
__buildid:
