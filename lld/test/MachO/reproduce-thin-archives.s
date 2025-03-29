# REQUIRES: x86

# RUN: rm -rf %t.dir; split-file %s %t.dir

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t.dir/foo.s -o %t.dir/foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %t.dir/unused.s -o %t.dir/unused.o
# RUN: cd %t.dir
# RUN: llvm-ar rcsT foo.a foo.o unused.o

# RUN: %lld foo.a -o /dev/null --reproduce repro.tar
# RUN: tar tf repro.tar | FileCheck -DPATH='repro/%:t.dir' %s

# RUN: %lld -all_load foo.a -o /dev/null --reproduce repro2.tar
# RUN: tar tf repro2.tar | FileCheck -DPATH='repro2/%:t.dir' %s

# RUN: %lld -ObjC foo.a -o /dev/null --reproduce repro3.tar
# RUN: tar tf repro3.tar | FileCheck -DPATH='repro3/%:t.dir' %s

# CHECK-DAG: [[PATH]]/foo.a
# CHECK-DAG: [[PATH]]/foo.o
# CHECK-DAG: [[PATH]]/unused.o

#--- foo.s
.globl _main
_main:
  nop

#--- unused.s
.globl _unused
_unused:
  nop
