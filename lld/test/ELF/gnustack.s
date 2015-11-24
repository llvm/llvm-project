# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux /dev/null -o %t2
# RUN: ld.lld %t1 -z execstack -o %t
# RUN: llvm-readobj --program-headers -s %t | FileCheck --check-prefix=CHECK_RWX %s
# RUN: ld.lld %t1 -o %t
# RUN: llvm-readobj --program-headers -s %t | FileCheck --check-prefix=CHECK_RW %s
# RUN: ld.lld %t1 %t2 -o %t
# RUN: llvm-readobj --program-headers -s %t | FileCheck --check-prefix=CHECK_RWX %s
# RUN: ld.lld %t1 %t2 -z noexecstack -o %t
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

# CHECK_RWX-NOT: Type: PT_GNU_STACK

.globl _start
_start:
.section .note.GNU-stack,""
