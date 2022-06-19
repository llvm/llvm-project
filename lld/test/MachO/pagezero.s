# REQUIRES: x86, aarch64
# RUN: rm -rf %t; mkdir %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/x86_64.o
# RUN: llvm-mc -filetype=obj -triple=arm64_32-apple-darwin %s -o %t/arm64_32.o

# RUN: %lld -lSystem -arch x86_64 -o %t/x86_64 %t/x86_64.o -pagezero_size 100000
# RUN: llvm-readobj --macho-segment %t/x86_64 | FileCheck %s -D#VMSIZE=0x100000 -D#SIZE=72

# RUN: %lld-watchos -lSystem -arch arm64_32 -o %t/arm64_32 %t/arm64_32.o -pagezero_size 100000
# RUN: llvm-readobj --macho-segment %t/arm64_32 | FileCheck %s -D#VMSIZE=0x100000 -D#SIZE=56

# RUN: %lld -lSystem -arch x86_64 -o %t/zero %t/x86_64.o -pagezero_size 0
# RUN: llvm-readobj --macho-segment %t/zero | FileCheck %s --check-prefix=CHECK-ZERO -D#VMSIZE=0x1000 -D#SIZE=152

# RUN: %no-fatal-warnings-lld -lSystem -arch x86_64 -o %t/x86_64-misalign %t/x86_64.o -pagezero_size 1001 2>&1 | FileCheck %s --check-prefix=LINK -D#SIZE=0x1000
# RUN: llvm-readobj --macho-segment %t/x86_64-misalign | FileCheck %s -D#VMSIZE=0x1000 -D#SIZE=72

# RUN: %no-fatal-warnings-lld-watchos -lSystem -arch arm64_32 -o %t/arm64_32-misalign-4K %t/arm64_32.o -pagezero_size 1001 2>&1 | FileCheck %s --check-prefix=LINK -D#SIZE=0x0
# RUN: llvm-readobj --macho-segment %t/arm64_32-misalign-4K | FileCheck %s --check-prefix=CHECK-ZERO -D#VMSIZE=0x4000 -D#SIZE=124

# RUN: %no-fatal-warnings-lld-watchos -lSystem -arch arm64_32 -o %t/arm64_32-misalign-16K %t/arm64_32.o -pagezero_size 4001 2>&1 | FileCheck %s --check-prefix=LINK -D#SIZE=0x4000
# RUN: llvm-readobj --macho-segment %t/arm64_32-misalign-16K | FileCheck %s -D#VMSIZE=0x4000 -D#SIZE=56

# LINK: warning: __PAGEZERO size is not page aligned, rounding down to 0x[[#%x,SIZE]]

# CHECK:        Name: __PAGEZERO
# CHECK-NEXT:   Size: [[#%d,SIZE]]
# CHECK-NEXT:   vmaddr: 0x0
# CHECK-NEXT:   vmsize: 0x[[#%x,VMSIZE]]

# CHECK-ZERO:        Name: __TEXT
# CHECK-ZERO-NEXT:   Size: [[#%d,SIZE]]
# CHECK-ZERO-NEXT:   vmaddr: 0x0
# CHECK-ZERO-NEXT:   vmsize: 0x[[#%x,VMSIZE]]

.globl _main
_main:
