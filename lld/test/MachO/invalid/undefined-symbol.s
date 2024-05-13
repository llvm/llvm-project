# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/main.s -o %t/main.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-ar crs %t/foo.a %t/foo.o
# RUN: not %lld --icf=all -o /dev/null %t/main.o 2>&1 | \
# RUN:     FileCheck %s -DSYM=__ZN3foo3barEi -DLOC='%t/main.o:(symbol _main+0x1)'
# RUN: not %lld -demangle --icf=all -o /dev/null %t/main.o 2>&1 | \
# RUN:     FileCheck %s -DSYM='foo::bar(int)' -DLOC='%t/main.o:(symbol main+0x1)'
# RUN: not %lld -o /dev/null %t/main.o %t/foo.a 2>&1 | \
# RUN:     FileCheck %s -DSYM=_bar -DLOC='%t/foo.a(foo.o):(symbol __ZN3foo3barEi+0x1)'
# RUN: not %lld -demangle -o /dev/null %t/main.o %t/foo.a 2>&1 | \
# RUN:     FileCheck %s -DSYM=bar -DLOC='%t/foo.a(foo.o):(symbol foo::bar(int)+0x1)'
# RUN: not %lld -o /dev/null %t/main.o -force_load %t/foo.a 2>&1 | \
# RUN:     FileCheck %s -DSYM=_bar -DLOC='%t/foo.a(foo.o):(symbol __ZN3foo3barEi+0x1)'
# RUN: not %lld -demangle -o /dev/null %t/main.o -force_load %t/foo.a 2>&1 | \
# RUN:     FileCheck %s -DSYM=bar -DLOC='%t/foo.a(foo.o):(symbol foo::bar(int)+0x1)'
# CHECK: error: undefined symbol: [[SYM]]
# CHECK-NEXT: >>> referenced by [[LOC]]

#--- foo.s
.globl __ZN3foo3barEi
.text
__ZN3foo3barEi:
  callq _bar
  retq

#--- main.s
.text

_anotherref:
  callq __ZN3foo3barEi
  movq $0, %rax
  retq

.globl _main
_main:
  callq __ZN3foo3barEi
  movq $0, %rax
  retq

.subsections_via_symbols
