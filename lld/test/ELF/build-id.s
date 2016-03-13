# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: ld.lld --build-id %t -o %t2
# RUN: llvm-objdump -s %t2 | FileCheck -check-prefix=BUILDID %s
# RUN: ld.lld %t -o %t2
# RUN: llvm-objdump -s %t2 | FileCheck -check-prefix=NO-BUILDID %s

.globl _start;
_start:
  nop

.section .note.test, "a", @note
   .quad 42

# BUILDID:      Contents of section .note.gnu.build-id:
# BUILDID-NEXT: 04000000 08000000 03000000 474e5500  ............GNU.
# BUILDID:      Contents of section .note.test:

# NO-BUILDID-NOT: Contents of section .note.gnu.build-id:
