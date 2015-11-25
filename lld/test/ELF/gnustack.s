# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1
# RUN: ld.lld %t1 -z execstack -o %t
# RUN: llvm-readobj --program-headers -s %t | FileCheck --check-prefix=RWX %s
# RUN: ld.lld %t1 -o %t
# RUN: llvm-readobj --program-headers -s %t | FileCheck --check-prefix=RW %s

# RW:       Sections [
# RW-NOT:   Name: .note.GNU-stack
# RW:       ProgramHeaders [
# RW:        ProgramHeader {
# RW:        Type: PT_GNU_STACK
# RW-NEXT:   Offset: 0x0
# RW-NEXT:   VirtualAddress: 0x0
# RW-NEXT:   PhysicalAddress: 0x0
# RW-NEXT:   FileSize: 0
# RW-NEXT:   MemSize: 0
# RW-NEXT:   Flags [
# RW-NEXT:     PF_R
# RW-NEXT:     PF_W
# RW-NEXT:   ]
# RW-NEXT:   Alignment: 0
# RW-NEXT:   }
# RW-NEXT: ]

# RWX-NOT: Name: .note.GNU-stack
# RWX-NOT: Type: PT_GNU_STACK

.globl _start
_start:
