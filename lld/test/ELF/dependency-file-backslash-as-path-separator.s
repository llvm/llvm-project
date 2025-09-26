# REQUIRES: x86, system-windows
# RUN: mkdir -p %t/back
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o "%t/back\\slash.o"
# RUN: ld.lld -o %t/foo.exe "%t/back\\slash.o" --dependency-file=%t/foo.d
# RUN: FileCheck --match-full-lines -DFILE=%t %s < %t/foo.d

# CHECK:      [[FILE]]\foo.exe: \
# CHECK-NEXT:   [[FILE]]\back\slash.o
# CHECK-EMPTY:
# CHECK-NEXT: [[FILE]]\back\slash.o:

.global _start
_start:
