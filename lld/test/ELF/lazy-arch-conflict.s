# REQUIRES: x86

# RUN: echo '.weak foo; .quad foo;' | llvm-mc -filetype=obj -triple=x86_64 - -o %t64.o
# RUN: echo '.globl foo; foo:' | llvm-mc -filetype=obj -triple=i686-pc-linux - -o %t32.o
# RUN: not ld.lld %t64.o --start-lib %t32.o --end-lib -o /dev/null 2>&1 | FileCheck %s

# CHECK: error: {{.*}}32.o is incompatible with {{.*}}64.o
