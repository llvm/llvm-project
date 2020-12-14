# REQUIRES: x86
# RUN: rm -rf %t
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/main.s -o %t/main.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-ar crs %t/foo.a %t/foo.o
# RUN: not %lld -o /dev/null %t/main.o 2>&1 | \
# RUN:     FileCheck %s -DSYM=_foo -DFILENAME=%t/main.o
# RUN: not %lld -o /dev/null %t/main.o %t/foo.a 2>&1 | \
# RUN:     FileCheck %s -DSYM=_bar -DFILENAME='foo.a(foo.o)'
# RUN: not %lld -o /dev/null %t/main.o -force_load %t/foo.a 2>&1 | \
# RUN:     FileCheck %s -DSYM=_bar -DFILENAME='foo.a(foo.o)'
# CHECK: error: undefined symbol: [[SYM]]
# CHECK-NEXT: >>> referenced by [[FILENAME]]

#--- foo.s
.globl _foo
.text
_foo:
  callq _bar
  retq

#--- main.s
.globl _main
.text
_main:
  callq _foo
  movq $0, %rax
  retq
