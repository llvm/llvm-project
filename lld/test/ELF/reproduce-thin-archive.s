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

## With multiple thin archives + --threads>1, ensure deterministic member order.
# RUN: llvm-mc -filetype=obj -triple=x86_64 /dev/null -o n.o
# RUN: cp n.o b.o && cp n.o c.o && cp n.o d.o
# RUN: llvm-ar --format=gnu rcT b.a b.o
# RUN: llvm-ar --format=gnu rcT c.a c.o
# RUN: llvm-ar --format=gnu rcT d.a d.o
# RUN: ld.lld foo.a b.a --whole-archive c.a d.a --reproduce repro2.tar
# RUN: tar tf repro2.tar | FileCheck -DPATH='repro2/%:t.dir' --check-prefix=CHECK2 %s

# CHECK2:      [[PATH]]/foo.a
# CHECK2-NEXT: [[PATH]]/b.a
# CHECK2-NEXT: [[PATH]]/c.a
# CHECK2-NEXT: [[PATH]]/d.a
# CHECK2-NEXT: [[PATH]]/foo.o
# CHECK2-NEXT: [[PATH]]/b.o
# CHECK2-NEXT: [[PATH]]/c.o
# CHECK2-NEXT: [[PATH]]/d.o

.globl _start
_start:
  nop
