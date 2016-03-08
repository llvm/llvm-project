# REQUIRES: x86

# RUN: mkdir -p %t.dir/lib
# RUN: mkdir -p %t.dir/usr/lib
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/nostdlib.s -o %t2.o
# RUN: ld.lld -shared -o %t.dir/lib/libfoo.so %t2.o
# RUN: ld.lld -shared -o %t.dir/usr/lib/libbar.so %t2.o
# RUN: ld.lld --sysroot=%t.dir -o %t %t1.o -lfoo
# RUN: ld.lld --sysroot=%t.dir -o %t %t1.o -lbar
# RUN: not ld.lld --sysroot=%t.dir -nostdlib -o %t %t1.o -lfoo
# RUN: not ld.lld --sysroot=%t.dir -nostdlib -o %t %t1.o -lbar

.globl _start
_start:
  ret
