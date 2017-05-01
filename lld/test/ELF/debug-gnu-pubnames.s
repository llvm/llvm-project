# REQUIRES: x86
# RUN: ld.lld -e main %p/Inputs/gdb-index-a.elf %p/Inputs/gdb-index-b.elf -o %t1.exe
# RUN: llvm-readobj -sections %t1.exe | FileCheck -check-prefix=CHECK1 %s
# CHECK1: Name: .debug_gnu_pubnames
# CHECK1: Name: .debug_gnu_pubtypes

# RUN: ld.lld -gdb-index -e main %p/Inputs/gdb-index-a.elf %p/Inputs/gdb-index-b.elf -o %t2.exe
# RUN: llvm-readobj -sections %t2.exe | FileCheck -check-prefix=CHECK2 %s
# CHECK2-NOT: Name: .debug_gnu_pubnames
# CHECK2-NOT: Name: .debug_gnu_pubtypes
