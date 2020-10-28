# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: %lld -o %t %t.o
# RUN: llvm-readobj --section-headers --macho-segment %t | FileCheck %s

# CHECK:      Name: __text
# CHECK-NEXT: Segment: __TEXT
# CHECK-NOT:  }
# CHECK:      Alignment: 1
# CHECK-NOT:  }
# CHECK:      Type: Regular (0x0)
# CHECK-NEXT: Attributes [ (0x800004)
# CHECK-NEXT:   PureInstructions (0x800000)
# CHECK-NEXT:   SomeInstructions (0x4)
# CHECK-NEXT: ]

# CHECK:      Name: __cstring
# CHECK-NEXT: Segment: __TEXT
# CHECK-NOT:  }
# CHECK:      Alignment: 2
# CHECK-NOT:  }
# CHECK:      Type: CStringLiterals (0x2)
# CHECK-NEXT: Attributes [ (0x0)
# CHECK-NEXT: ]

# CHECK:      Name: maxlen_16ch_name
# CHECK-NEXT: Segment: __TEXT
# CHECK-NEXT: Address:
# CHECK-NEXT: Size: [[#%x, LAST_SEC_SIZE:]]
# CHECK-NEXT: Offset: [[#%u, LAST_SEC_OFF:]]
# CHECK-NEXT: Alignment: 3
# CHECK-NOT:  }
# CHECK:      Type: Regular (0x0)

# CHECK-LABEL: Segment {
# CHECK:       Name: __TEXT
# CHECK-NEXT:  Size:
# CHECK-NEXT:  vmaddr:
# CHECK-NEXT:  vmsize:
# CHECK-NEXT:  fileoff: 0
# CHECK-NEXT:  filesize: [[#%u, LAST_SEC_SIZE + LAST_SEC_OFF]]

.text
.align 1
.global _main
_main:
  mov $0, %rax
  ret

.section __TEXT,__cstring
.align 2
str:
  .asciz "Hello world!\n"

.section __TEXT,maxlen_16ch_name
.align 3
