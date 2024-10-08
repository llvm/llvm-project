# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t

# RUN: ld.lld --build-id %t -o %t2
# RUN: llvm-readobj -S %t2 | FileCheck -check-prefix=ALIGN %s

# RUN: ld.lld --build-id %t -o %t2
# RUN: llvm-objdump -s %t2 | FileCheck --check-prefix=SHA1 %s
# RUN: ld.lld --build-id %t -o %t2 --threads=1
# RUN: llvm-objdump -s %t2 | FileCheck --check-prefix=SHA1 %s

# RUN: ld.lld --build-id=fast %t -o %t2
# RUN: llvm-objdump -s %t2 | FileCheck --check-prefix=FAST %s

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
# RUN: llvm-objdump -s %t2 | FileCheck --check-prefix=SHA1 %s

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

# FAST:      Contents of section .note.test:
# FAST:      Contents of section .note.gnu.build-id:
# FAST-NEXT: 04000000 08000000 03000000 474e5500  ............GNU.
# FAST-NEXT: 630bc2f5 a2584763

# MD5:      Contents of section .note.gnu.build-id:
# MD5-NEXT: 04000000 10000000 03000000 474e5500  ............GNU.
# MD5-NEXT: dbf0bc13 b3ff11e9 fde6e17c 0304983c

# SHA1:      Contents of section .note.gnu.build-id:
# SHA1-NEXT: 04000000 14000000 03000000 474e5500  ............GNU.
# SHA1-NEXT: 1215775f d3b60050 70afd970 e8a10972

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
# SEPARATE-NEXT: 0x002001a8 5cd067a4 2631c0fd 42029037 4b8e0938

# RUN: ld.lld --build-id=sha1 --no-rosegment %t -o %t2
# RUN: llvm-readelf -x .note.gnu.build-id %t2 | FileCheck --check-prefix=NORO %s

# NORO:      Hex dump of section '.note.gnu.build-id':
# NORO-NEXT: 0x00200160 04000000 14000000 03000000 474e5500
# NORO-NEXT: 0x00200170 a328cc99 45bfc3fc a9fc8615 37102f9d
