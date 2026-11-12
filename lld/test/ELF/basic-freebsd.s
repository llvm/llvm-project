# REQUIRES: x86
# Verify that OSABI is set to the correct value.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 empty.s -o empty.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-freebsd a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-linux gnu.s -o gnu.o
# RUN: ld.lld a.o -o out
# RUN: llvm-readobj --file-headers out | FileCheck %s
# RUN: ld.lld empty.o a.o gnu.o empty.o -o out2
# RUN: llvm-readobj --file-headers out2 | FileCheck %s

#--- empty.s
#--- a.s
.globl _start
_start:
  mov $1, %rax
  mov $42, %rdi
  syscall
#--- gnu.s
.section retain,"aR"

# CHECK: ElfHeader {
# CHECK-NEXT:   Ident {
# CHECK-NEXT:     Magic: (7F 45 4C 46)
# CHECK-NEXT:     Class: 64-bit (0x2)
# CHECK-NEXT:     DataEncoding: LittleEndian (0x1)
# CHECK-NEXT:     FileVersion: 1
# CHECK-NEXT:     OS/ABI: FreeBSD (0x9)
# CHECK-NEXT:     ABIVersion: 0
# CHECK-NEXT:     Unused: (00 00 00 00 00 00 00)
# CHECK-NEXT:   }
# CHECK-NEXT:   Type: Executable (0x2)
# CHECK-NEXT:   Machine: EM_X86_64 (0x3E)
