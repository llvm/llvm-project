# REQUIRES: x86

# RUN: rm -rf %t.dir
# RUN: mkdir -p %t.dir
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %s -o %t.dir/foo.o
# RUN: cd %t.dir
# RUN: llvm-ar rcsT foo.a foo.o

# RUN: %lld foo.a -o /dev/null --reproduce repro.tar
# RUN: tar tf repro.tar | FileCheck -DPATH='repro/%:t.dir' %s

# RUN: %lld -all_load foo.a -o /dev/null --reproduce repro2.tar
# RUN: tar tf repro2.tar | FileCheck -DPATH='repro2/%:t.dir' %s

# CHECK-DAG: [[PATH]]/foo.a
# CHECK-DAG: [[PATH]]/foo.o

.globl _main
_main:
  nop
