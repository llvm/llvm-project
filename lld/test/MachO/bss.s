# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: %lld -o %t %t.o
# RUN: llvm-readobj --section-headers --macho-segment %t | FileCheck %s

## Check that __bss takes up zero file size, is at file offset zero, and appears
## at the end of its segment. Also check that __tbss is placed immediately
## before it.
## Zerofill sections in other segments (i.e. not __DATA) should also be placed
## at the end.

# CHECK:        Index: 1
# CHECK-NEXT:   Name: __data
# CHECK-NEXT:   Segment: __DATA
# CHECK-NEXT:   Address:
# CHECK-NEXT:   Size: 0x8
# CHECK-NEXT:   Offset: 4096
# CHECK-NEXT:   Alignment: 0
# CHECK-NEXT:   RelocationOffset: 0x0
# CHECK-NEXT:   RelocationCount: 0
# CHECK-NEXT:   Type: Regular (0x0)
# CHECK-NEXT:   Attributes [ (0x0)
# CHECK-NEXT:   ]
# CHECK-NEXT:   Reserved1: 0x0
# CHECK-NEXT:   Reserved2: 0x0
# CHECK-NEXT:   Reserved3: 0x0

# CHECK:        Index: 2
# CHECK-NEXT:   Name: __thread_bss
# CHECK-NEXT:   Segment: __DATA
# CHECK-NEXT:   Address:
# CHECK-NEXT:   Size: 0x4
# CHECK-NEXT:   Offset: 0
# CHECK-NEXT:   Alignment: 0
# CHECK-NEXT:   RelocationOffset: 0x0
# CHECK-NEXT:   RelocationCount: 0
# CHECK-NEXT:   Type: ThreadLocalZerofill (0x12)
# CHECK-NEXT:   Attributes [ (0x0)
# CHECK-NEXT:   ]
# CHECK-NEXT:   Reserved1: 0x0
# CHECK-NEXT:   Reserved2: 0x0
# CHECK-NEXT:   Reserved3: 0x0

# CHECK:        Index: 3
# CHECK-NEXT:   Name: __bss
# CHECK-NEXT:   Segment: __DATA
# CHECK-NEXT:   Address:
# CHECK-NEXT:   Size: 0x8
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

# CHECK:        Index: 4
# CHECK-NEXT:   Name: foo
# CHECK-NEXT:   Segment: FOO
# CHECK-NEXT:   Address:
# CHECK-NEXT:   Size: 0x8
# CHECK-NEXT:   Offset: 8192
# CHECK-NEXT:   Alignment: 0
# CHECK-NEXT:   RelocationOffset: 0x0
# CHECK-NEXT:   RelocationCount: 0
# CHECK-NEXT:   Type: Regular (0x0)
# CHECK-NEXT:   Attributes [ (0x0)
# CHECK-NEXT:   ]
# CHECK-NEXT:   Reserved1: 0x0
# CHECK-NEXT:   Reserved2: 0x0
# CHECK-NEXT:   Reserved3: 0x0

# CHECK:        Index: 5
# CHECK-NEXT:   Name: bss
# CHECK-NEXT:   Segment: FOO
# CHECK-NEXT:   Address:
# CHECK-NEXT:   Size: 0x8
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
# CHECK-NEXT: vmsize: 0x14
# CHECK-NEXT: fileoff:
# CHECK-NEXT: filesize: 8

# CHECK:      Name: FOO
# CHECK-NEXT: Size:
# CHECK-NEXT: vmaddr:
# CHECK-NEXT: vmsize: 0x10
# CHECK-NEXT: fileoff:
# CHECK-NEXT: filesize: 8

.globl _main

.text
_main:
  movq $0, %rax
  retq

.bss
.zero 4

.tbss _foo, 4
.zero 4

.data
.quad 0x1234

.zerofill FOO,bss,_zero_foo,0x8

.section FOO,foo
.quad 123
