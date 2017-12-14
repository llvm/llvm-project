# RUN: llvm-mc %s -o %t.o -filetype=obj -triple x86_64-pc-linux

# We used to crash on this
# RUN: not ld.lld %t.o %p/Inputs/local-symbol-in-dso.so -o %t 2>&1 | FileCheck %s
# CHECK: Found local symbol 'foo' in global part of symbol table in file {{.*}}local-symbol-in-dso.so
