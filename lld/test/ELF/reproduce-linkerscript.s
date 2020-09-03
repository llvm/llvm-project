# REQUIRES: x86

# RUN: rm -rf %t.dir
# RUN: mkdir -p %t.dir/build
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.dir/build/foo.o
# RUN: echo "INPUT(\"%t.dir/build/foo.o\")" > %t.dir/build/foo.script
# RUN: echo "INCLUDE \"%t.dir/build/bar.script\"" >> %t.dir/build/foo.script
# RUN: echo "/* empty */" > %t.dir/build/bar.script
# RUN: cd %t.dir
# RUN: ld.lld build/foo.script -o /dev/null --reproduce repro.tar
# RUN: tar tf repro.tar | FileCheck -DPATH='%:t.dir' %s

# CHECK: [[PATH]]/build/foo.script
# CHECK: [[PATH]]/build/foo.o
# CHECK: [[PATH]]/build/bar.script

.globl _start
_start:
  mov $60, %rax
  mov $42, %rdi
  syscall
