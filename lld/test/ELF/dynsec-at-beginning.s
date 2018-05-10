# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t

# RUN: ld.lld --hash-style=gnu -o %t1  %t -shared
# RUN: llvm-readobj -elf-output-style=GNU -s %t1 | FileCheck %s

# Dynamic symbol and dynamic strtab sections are at the beginning of
# SHF_ALLOC sections.
# CHECK:      .dynsym  {{.*}}   A
# CHECK-NEXT: .dynstr  {{.*}}   A
# CHECK-NEXT: foo      {{.*}}   A
# CHECK-NEXT: .hash    {{.*}}   A
# CHECK-NEXT: .text    {{.*}}   AX

.section foo, "a"
.byte 0
