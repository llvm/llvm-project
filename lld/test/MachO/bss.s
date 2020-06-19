# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: lld -flavor darwinnew -o %t %t.o
# RUN: llvm-readobj --section-headers --macho-segment %t | FileCheck %s

## Check that __bss takes up zero file size and is at file offset zero.

# CHECK:        Name: __bss
# CHECK-NEXT:   Segment: __DATA
# CHECK-NEXT:   Address:
# CHECK-NEXT:   Size: 0x4
# CHECK-NEXT:   Offset: 0
# CHECK-NEXT:   Alignment: 0
# CHECK-NEXT:   RelocationOffset: 0x0
# CHECK-NEXT:   RelocationCount: 0
# CHECK-NEXT:   Type: ZeroFill (0x1)
# CHECK-NEXT:   Attributes [ (0x0)
# CHECK-NEXT:   ]
# CHECK-NEXT:   Reserved1: 0x0
# CHECK-NEXT:   Reserved2: 0x0
# CHECK-NEXT:   Reserved3: 0x0

# CHECK:      Name: __DATA
# CHECK-NEXT: Size:
# CHECK-NEXT: vmaddr:
# CHECK-NEXT: vmsize: 0x4
# CHECK-NEXT: fileoff:
# CHECK-NEXT: filesize: 0

.globl _main

.text
_main:
  movq $0, %rax
  retq

.bss
.zero 4
