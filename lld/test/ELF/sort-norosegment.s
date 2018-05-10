# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t

# RUN: ld.lld --hash-style=sysv -no-rosegment -o %t1  %t -shared
# RUN: llvm-readobj -elf-output-style=GNU -s %t1 | FileCheck %s

# CHECK:      .dynsym  {{.*}}   A
# CHECK-NEXT: .dynstr  {{.*}}   A
# CHECK-NEXT: .text    {{.*}}   AX
# CHECK-NEXT: .hash    {{.*}}   A
# CHECK-NEXT: foo      {{.*}}  WA
# CHECK-NEXT: .dynamic {{.*}}  WA

.section foo, "aw"
.byte 0
