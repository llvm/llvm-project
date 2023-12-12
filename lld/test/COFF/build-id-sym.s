# REQUIRES: x86
# RUN: llvm-mc -triple=x86_64-windows-msvc -filetype=obj -o %t.obj %s
# RUN: lld-link -debug:symtab -entry:main %t.obj -build-id -Brepro -out:%t.exe
# RUN: llvm-objdump -s -t %t.exe | FileCheck %s

# Check __build_guid points to 0x140002038 which is the start of build id.

# CHECK:      SYMBOL TABLE:
# CHECK-NEXT: 0x00000038 __build_guid
# CHECK:      Contents of section .rdata:
# CHECK-NEXT:  140002000 00000000 dc1b27c9 00000000 02000000  ......'.........
# CHECK-NEXT:  140002010 19000000 38200000 38060000 00000000  ....8 ..8.......
# CHECK-NEXT:  140002020 dc1b27c9 00000000 10000000 00000000  ..'.............
# CHECK-NEXT:  140002030 00000000 00000000 52534453 dc1b27c9  ........RSDS..'.
# CHECK-NEXT:  140002040 eef6f7b6 4c4c4420 5044422e 01000000  ....LLD PDB.....

.globl main
main:
  nop

.section .bss,"bw",discard,__build_guid
.global __build_guid
__build_guid:
