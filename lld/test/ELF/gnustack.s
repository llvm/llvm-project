# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1
# RUN: ld.lld2 %t1 -z execstack -o %t
# RUN: llvm-readobj --program-headers -s %t | FileCheck --check-prefix=CHECK_RWX %s
# RUN: ld.lld2 %t1 -o %t
# RUN: llvm-readobj --program-headers -s %t | FileCheck --check-prefix=CHECK_RW %s

# CHECK_RW:       Sections [
# CHECK_RW-NOT:   Name: .note.GNU-stack
# CHECK_RW:       ProgramHeaders [
# CHECK_RW:        ProgramHeader {
# CHECK_RW:        Type: PT_GNU_STACK
# CHECK_RW-NEXT:   Offset: 0x0
# CHECK_RW-NEXT:   VirtualAddress: 0x0
# CHECK_RW-NEXT:   PhysicalAddress: 0x0
# CHECK_RW-NEXT:   FileSize: 0
# CHECK_RW-NEXT:   MemSize: 0
# CHECK_RW-NEXT:   Flags [
# CHECK_RW-NEXT:     PF_R
# CHECK_RW-NEXT:     PF_W
# CHECK_RW-NEXT:   ]
# CHECK_RW-NEXT:   Alignment: 0
# CHECK_RW-NEXT:   }
# CHECK_RW-NEXT: ]

# CHECK_RWX:       Sections [
# CHECK_RWX-NOT:   Name: .note.GNU-stack
# CHECK_RWX:       ProgramHeaders [
# CHECK_RWX:        ProgramHeader {
# CHECK_RWX:        Type: PT_GNU_STACK
# CHECK_RWX-NEXT:   Offset: 0x0
# CHECK_RWX-NEXT:   VirtualAddress: 0x0
# CHECK_RWX-NEXT:   PhysicalAddress: 0x0
# CHECK_RWX-NEXT:   FileSize: 0
# CHECK_RWX-NEXT:   MemSize: 0
# CHECK_RWX-NEXT:   Flags [
# CHECK_RWX-NEXT:     PF_R
# CHECK_RWX-NEXT:     PF_W
# CHECK_RWX-NEXT:     PF_X
# CHECK_RWX-NEXT:   ]
# CHECK_RWX-NEXT:   Alignment: 0
# CHECK_RWX-NEXT:   }
# CHECK_RWX-NEXT: ]

.globl _start
_start:
