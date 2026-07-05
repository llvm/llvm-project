# REQUIRES: riscv
## Don't forget to update st_value(foo) when foo is defined in another relocatable object file.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax b.s -o b.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax w.s -o w.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax w2.s -o w2.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax call_foo.s -o call_foo.o

# RUN: ld.lld -r b.o w.o -o bw.o
# RUN: ld.lld -Ttext=0x10000 a.o bw.o --wrap=foo -o 1
# RUN: llvm-objdump -d --no-show-raw-insn 1 | FileCheck %s

# RUN: ld.lld -r a.o b.o -o ab.o
# RUN: ld.lld -Ttext=0x10000 ab.o w.o --wrap=foo -o 2
# RUN: llvm-objdump -d --no-show-raw-insn 2 | FileCheck %s

# RUN: ld.lld -Ttext=0x10000 w2.o call_foo.o --wrap=foo -o 3
# RUN: llvm-objdump -d --no-show-raw-insn 3 | FileCheck %s --check-prefix=CHECK2

# CHECK-LABEL:  <_start>:
# CHECK-NEXT:     10000: jal {{.*}} <__wrap_foo>
# CHECK-EMPTY:
# CHECK-NEXT:   <foo>:
# CHECK-NEXT:     10004: jal {{.*}} <__wrap_foo>
# CHECK-EMPTY:
# CHECK-NEXT:   <__wrap_foo>:
# CHECK-NEXT:     10008: jal {{.*}} <foo>

# CHECK2-LABEL: <_start>:
# CHECK2-NEXT:    jal {{.*}} <call_foo>
# CHECK2-EMPTY:
# CHECK2-NEXT:  <__wrap_foo>:
# CHECK2-NEXT:    ret
# CHECK2-EMPTY:
# CHECK2-NEXT:  <call_foo>:
# CHECK2-NEXT:    jal {{.*}} <__wrap_foo>

#--- a.s
.globl _start
_start:
  call foo

#--- b.s
.globl foo
foo:
  call __wrap_foo

#--- w.s
.globl __wrap_foo
__wrap_foo:
  call __real_foo

#--- w2.s
.globl _start, __wrap_foo
_start:
  call call_foo

__wrap_foo:
  ret

#--- call_foo.s
.globl call_foo
call_foo:
  call foo
