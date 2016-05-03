# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: ld.lld --build-id %t -o %t2
# RUN: llvm-objdump -s %t2 | FileCheck -check-prefix=DEFAULT %s
# RUN: ld.lld --build-id=md5 %t -o %t2
# RUN: llvm-objdump -s %t2 | FileCheck -check-prefix=MD5 %s
# RUN: ld.lld --build-id=sha1 %t -o %t2
# RUN: llvm-objdump -s %t2 | FileCheck -check-prefix=SHA1 %s
# RUN: ld.lld %t -o %t2
# RUN: llvm-objdump -s %t2 | FileCheck -check-prefix=NONE %s
# RUN: ld.lld --build-id=md5 --build-id=none %t -o %t2
# RUN: llvm-objdump -s %t2 | FileCheck -check-prefix=NONE %s

.globl _start
_start:
  nop

.section .note.test, "a", @note
   .quad 42

# DEFAULT:      Contents of section .note.gnu.build-id:
# DEFAULT-NEXT: 04000000 08000000 03000000 474e5500  ............GNU.
# DEFAULT:      Contents of section .note.test:

# MD5:      Contents of section .note.gnu.build-id:
# MD5-NEXT: 04000000 10000000 03000000 474e5500  ............GNU.

# SHA1:      Contents of section .note.gnu.build-id:
# SHA1-NEXT: 04000000 14000000 03000000 474e5500  ............GNU.

# NONE-NOT: Contents of section .note.gnu.build-id:
