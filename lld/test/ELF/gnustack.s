# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 x.s -o x.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 nox.s -o nox.o

# RUN: ld.lld a.o -z execstack -o out
# RUN: llvm-readobj --program-headers -S out | FileCheck --check-prefix=RWX %s

# RUN: ld.lld a.o -o out
# RUN: llvm-readobj --program-headers -S out | FileCheck --check-prefix=RW %s

# RUN: ld.lld a.o -o out -z noexecstack
# RUN: llvm-readobj --program-headers -S out | FileCheck --check-prefix=RW %s

# RUN: ld.lld a.o -o out -z nognustack
# RUN: llvm-readobj --program-headers -s out | FileCheck --check-prefix=NOGNUSTACK %s

# RW:      Type: PT_GNU_STACK
# RW-NEXT: Offset: 0x0
# RW-NEXT: VirtualAddress: 0x0
# RW-NEXT: PhysicalAddress: 0x0
# RW-NEXT: FileSize: 0
# RW-NEXT: MemSize: 0
# RW-NEXT: Flags [
# RW-NEXT:   PF_R
# RW-NEXT:   PF_W
# RW-NEXT: ]
# RW-NEXT: Alignment: 0

# RWX:      Type: PT_GNU_STACK
# RWX-NEXT: Offset: 0x0
# RWX-NEXT: VirtualAddress: 0x0
# RWX-NEXT: PhysicalAddress: 0x0
# RWX-NEXT: FileSize: 0
# RWX-NEXT: MemSize: 0
# RWX-NEXT: Flags [
# RWX-NEXT:   PF_R
# RWX-NEXT:   PF_W
# RWX-NEXT:   PF_X
# RWX-NEXT: ]
# RWX-NEXT: Alignment: 0

# NOGNUSTACK-NOT: Type: PT_GNU_STACK

# RUN: not ld.lld a.o x.o nox.o x.o 2>&1 | FileCheck %s --check-prefix=ERR --implicit-check-not=error:
# RUN: not ld.lld a.o x.o nox.o x.o -z nognustack 2>&1 | FileCheck %s --check-prefix=ERR --implicit-check-not=error:
# ERR-COUNT-2: error: x.o: requires an executable stack, but -z execstack is not specified

# RUN: ld.lld a.o x.o nox.o x.o -z execstack --fatal-warnings
# RUN: ld.lld -r x.o --fatal-warnings

#--- a.s
.globl _start
_start:

#--- x.s
.section .note.GNU-stack,"x"

#--- nox.s
.section .note.GNU-stack,""
