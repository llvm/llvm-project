# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: not ld.lld %t.o -o %t.so -shared 2>&1 | FileCheck %s

# CHECK: undefined symbol: hidden in {{.*}}
.global hidden
.hidden hidden

# CHECK: undefined symbol: internal in {{.*}}
.global internal
.internal internal

# CHECK: undefined symbol: protected in {{.*}}
.global protected
.protected protected
