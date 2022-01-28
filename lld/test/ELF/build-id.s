# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t

# RUN: ld.lld --build-id %t -o %t2
# RUN: llvm-readobj -S %t2 | FileCheck -check-prefix=ALIGN %s

# RUN: ld.lld --build-id %t -o %t2
# RUN: llvm-objdump -s %t2 | FileCheck --check-prefix=DEFAULT %s
# RUN: ld.lld --build-id=fast %t -o %t2
# RUN: llvm-objdump -s %t2 | FileCheck --check-prefix=DEFAULT %s
# RUN: ld.lld --build-id %t -o %t2 --threads=1
# RUN: llvm-objdump -s %t2 | FileCheck --check-prefix=DEFAULT %s

# RUN: ld.lld --build-id=md5 %t -o %t2
# RUN: llvm-objdump -s %t2 | FileCheck --check-prefix=MD5 %s
# RUN: ld.lld --build-id=md5 %t -o %t2 --threads=1
# RUN: llvm-objdump -s %t2 | FileCheck --check-prefix=MD5 %s

# RUN: ld.lld --build-id=sha1 %t -o %t2
# RUN: llvm-objdump -s %t2 | FileCheck --check-prefix=SHA1 %s
# RUN: ld.lld --build-id=sha1 %t -o %t2 --threads=1
# RUN: llvm-objdump -s %t2 | FileCheck --check-prefix=SHA1 %s

# RUN: ld.lld --build-id=tree %t -o %t2
# RUN: llvm-objdump -s %t2 | FileCheck --check-prefix=SHA1 %s
# RUN: ld.lld --build-id=tree %t -o %t2 --threads=1
# RUN: llvm-objdump -s %t2 | FileCheck --check-prefix=SHA1 %s

# RUN: ld.lld --build-id=uuid %t -o %t2
# RUN: llvm-objdump -s %t2 | FileCheck --check-prefix=UUID %s

# RUN: ld.lld --build-id=0x12345678 %t -o %t2
# RUN: llvm-objdump -s %t2 | FileCheck --check-prefix=HEX %s

# RUN: ld.lld %t -o %t2
# RUN: llvm-objdump -s %t2 | FileCheck --check-prefix=NONE %s

# RUN: ld.lld --build-id=md5 --build-id=none %t -o %t2
# RUN: llvm-objdump -s %t2 | FileCheck --check-prefix=NONE %s
# RUN: ld.lld --build-id --build-id=none %t -o %t2
# RUN: llvm-objdump -s %t2 | FileCheck --check-prefix=NONE %s
# RUN: ld.lld --build-id=none --build-id %t -o %t2
# RUN: llvm-objdump -s %t2 | FileCheck --check-prefix=DEFAULT %s

.globl _start
_start:
  nop

.section .note.test, "a", @note
   .quad 42

# ALIGN:      Name: .note.gnu.build-id
# ALIGN-NEXT: Type: SHT_NOTE
# ALIGN-NEXT: Flags [
# ALIGN-NEXT:   SHF_ALLOC
# ALIGN-NEXT: ]
# ALIGN-NEXT: Address:
# ALIGN-NEXT: Offset: [[_:0x[0-9A-Z]*(0|4|8|C)$]]
# ALIGN-NEXT: Size:
# ALIGN-NEXT: Link:
# ALIGN-NEXT: Info:
# ALIGN-NEXT: AddressAlignment: 4

# DEFAULT:      Contents of section .note.test:
# DEFAULT:      Contents of section .note.gnu.build-id:
# DEFAULT-NEXT: 04000000 08000000 03000000 474e5500  ............GNU.
# DEFAULT-NEXT: 7e8ddeff 3ed41fa3

# MD5:      Contents of section .note.gnu.build-id:
# MD5-NEXT: 04000000 10000000 03000000 474e5500  ............GNU.
# MD5-NEXT: 7b00fd9e 054ceb4b 06f64d0e 482cb476

# SHA1:      Contents of section .note.gnu.build-id:
# SHA1-NEXT: 04000000 14000000 03000000 474e5500  ............GNU.
# SHA1-NEXT: 221a99da dd1d2bf3 05e48a91 dde8a0cb

# UUID:      Contents of section .note.gnu.build-id:
# UUID-NEXT: 04000000 10000000 03000000 474e5500  ............GNU.

# HEX:      Contents of section .note.gnu.build-id:
# HEX-NEXT: 04000000 04000000 03000000 474e5500  ............GNU.
# HEX-NEXT: 12345678

# NONE-NOT: Contents of section .note.gnu.build-id:

# RUN: ld.lld --build-id=sha1 -z separate-loadable-segments %t -o %t2
# RUN: llvm-readelf -x .note.gnu.build-id %t2 | FileCheck --check-prefix=SEPARATE %s

# SEPARATE:      Hex dump of section '.note.gnu.build-id':
# SEPARATE-NEXT: 0x00200198 04000000 14000000 03000000 474e5500
# SEPARATE-NEXT: 0x002001a8 96820adf d90d5470 0a0c32ff a88c4017

# RUN: ld.lld --build-id=sha1 --no-rosegment %t -o %t2
# RUN: llvm-readelf -x .note.gnu.build-id %t2 | FileCheck --check-prefix=NORO %s

# NORO:      Hex dump of section '.note.gnu.build-id':
# NORO-NEXT: 0x00200160 04000000 14000000 03000000 474e5500
# NORO-NEXT: 0x00200170 cf6d7b3a 0b3297c3 5b47c079 ce048349
