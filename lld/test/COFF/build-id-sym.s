# REQUIRES: x86
# RUN: llvm-mc -triple=x86_64-windows-msvc -filetype=obj -o %t.obj %s
# RUN: lld-link -entry:main %t.obj -build-id -Brepro -out:%t.exe
# RUN: llvm-objdump -s %t.exe | FileCheck %s

# Check __lld_buildid points to 0x140002038 which is the start of build id.

# CHECK:      Contents of section .rdata:
# CHECK-NEXT:  140002000 00000000 8064d9b6 00000000 02000000  .....d..........
# CHECK-NEXT:  140002010 19000000 38200000 38060000 00000000  ....8 ..8.......
# CHECK-NEXT:  140002020 8064d9b6 00000000 10000000 00000000  .d..............
# CHECK-NEXT:  140002030 00000000 00000000 52534453 8064d9b6  ........RSDS.d..

# CHECK:      Contents of section .data:
# CHECK-NEXT:  140003000 38200040 01000000

.globl main
main:
  nop

.data
  .quad __lld_buildid
