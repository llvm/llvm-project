# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1.o
# RUN: ld.lld --build-id=0xcafebabe -o %t2.o %t1.o -r
# RUN: ld.lld --build-id=0xdeadbeef -o %t.exe %t2.o
# RUN: llvm-objdump -s %t.exe | FileCheck %s

## The default --build-id=none removes .note.gnu.build-id input sections.
# RUN: ld.lld %t2.o -o %t.none
# RUN: llvm-readelf -S %t.none | FileCheck %s --check-prefix=NO
# RUN: ld.lld --build-id=none %t2.o -o %t.none2
# RUN: cmp %t.none %t.none2

# CHECK: Contents of section .note.gnu.build-id:
# CHECK-NOT: cafebabe
# CHECK: deadbeef

# NO:     Section Headers:
# NO-NOT: .note.gnu.build-id

.global _start
_start:
  ret
