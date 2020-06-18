# REQUIRES: x86

# RUN: rm -rf %t.dir
# RUN: mkdir -p %t.dir
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.dir/foo.o
# RUN: cd %t.dir
# RUN: llvm-ar --format=gnu rcT foo.a foo.o

# RUN: ld.lld -m elf_x86_64 foo.a -o /dev/null --reproduce repro.tar
# RUN: tar tf repro.tar | FileCheck -DPATH='repro/%:t.dir' %s

# CHECK: [[PATH]]/foo.a
# CHECK: [[PATH]]/foo.o

# RUN: ld.lld -m elf_x86_64 --whole-archive foo.a -o /dev/null --reproduce repro2.tar
# RUN: tar tf repro2.tar | FileCheck -DPATH='repro2/%:t.dir' --check-prefix=CHECK2 %s

# CHECK2: [[PATH]]/foo.a
# CHECK2: [[PATH]]/foo.o

.globl _start
_start:
  nop
