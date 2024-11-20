# REQUIRES: x86
# RUN: llvm-mc -triple=x86_64-windows-msvc -filetype=obj -o %t.obj %s
# RUN: lld-link -debug:symtab -entry:main %t.obj -build-id -Brepro -out:%t.exe
# RUN: llvm-objdump -s -t %t.exe | FileCheck %s

# Check __buildid points to 0x14000203c which is after the signature RSDS.

# CHECK:      SYMBOL TABLE:
# CHECK-NEXT: 0x0000003c __buildid
# CHECK:      Contents of section .rdata:
# CHECK-NEXT:  140002000
# CHECK-NEXT:  140002010
# CHECK-NEXT:  140002020
# CHECK-NEXT:  140002030 {{.*}} {{.*}} 52534453 {{.*}}
# CHECK-NEXT:  140002040

.globl main
main:
  nop

.section .bss,"bw",discard,__buildid
.global __buildid
__buildid:
