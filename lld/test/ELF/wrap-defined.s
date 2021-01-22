# REQUIRES: x86

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/main.s -o %t/main.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/wrap.s -o %t/wrap.o
# RUN: ld.lld -shared --soname=fixed %t/wrap.o -o %t/wrap.so

## GNU ld does not wrap a defined symbol in an object file
## https://sourceware.org/bugzilla/show_bug.cgi?id=26358
## We choose to wrap defined symbols so that LTO, non-LTO and relocatable links
## behave the same. The 'call bar' in main.o will reference __wrap_bar. We cannot
## easily distinguish the case from cases where bar is not referenced, so we
## export __wrap_bar whenever bar is defined, regardless of whether it is indeed
## referenced.

# RUN: ld.lld -shared %t/main.o --wrap bar -o %t1.so
# RUN: llvm-objdump -d %t1.so | FileCheck %s
# RUN: ld.lld %t/main.o %t/wrap.so --wrap bar -o %t1
# RUN: llvm-objdump -d %t1 | FileCheck %s

# CHECK:      <_start>:
# CHECK-NEXT:   callq {{.*}} <__wrap_bar@plt>

#--- main.s
.globl _start, bar
_start:
  call bar
bar:

#--- wrap.s
.globl __wrap_bar
__wrap_bar:
  retq
