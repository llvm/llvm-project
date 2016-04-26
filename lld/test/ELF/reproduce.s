# REQUIRES: x86

# RUN: rm -rf %t.dir
# RUN: mkdir -p %t.dir/build1
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.dir/build1/foo.o
# RUN: cd %t.dir
# RUN: ld.lld build1/foo.o -o bar -shared --as-needed --reproduce repro
# RUN: diff build1/foo.o repro/build1/foo.o

# RUN: FileCheck %s --check-prefix=INVOCATION < repro/invocation.txt
# INVOCATION: lld{{[^\s]*}} build1/foo.o -o bar -shared --as-needed --reproduce repro

# RUN: mkdir -p %t.dir/build2/a/b/c
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.dir/build2/foo.o
# RUN: cd %t.dir/build2/a/b/c
# RUN: ld.lld ./../../../foo.o -o bar -shared --as-needed --reproduce repro
# RUN: diff %t.dir/build2/foo.o repro/foo.o

# RUN: not ld.lld build1/foo.o -o bar -shared --as-needed --reproduce . 2>&1 \
# RUN:   | FileCheck --check-prefix=ERROR %s
# ERROR: can't create directory

.globl _start
_start:
  mov $60, %rax
  mov $42, %rdi
  syscall
