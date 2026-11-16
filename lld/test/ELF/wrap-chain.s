# REQUIRES: x86
## With --wrap=foo and --wrap=__wrap_foo in either order, both object file
## references and --defsym resolve foo to __wrap_foo.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld %t.o -Ttext=0x201000 --defsym sym_foo=foo --wrap=__wrap_foo --wrap=foo -o %t1
# RUN: llvm-objdump -td --no-show-raw-insn %t1 | FileCheck %s
# RUN: ld.lld %t.o -Ttext=0x201000 --defsym sym_foo=foo --wrap=foo --wrap=__wrap_foo -o %t2
# RUN: llvm-objdump -td --no-show-raw-insn %t2 | FileCheck %s

# CHECK: 0000000000201000 g  .text {{.*}} foo
# CHECK: 0000000000201001 g  .text {{.*}} __wrap_foo
# CHECK: 0000000000201002 g  .text {{.*}} __wrap___wrap_foo
# CHECK: 0000000000201001 g  .text {{.*}} sym_foo

# CHECK-LABEL: <_start>:
# CHECK-NEXT:    callq 0x201001 <sym_foo>
# CHECK-NEXT:    callq 0x201002 <__wrap___wrap_foo>
# CHECK-NEXT:    movl $0x201001, %eax

.globl _start, foo, __wrap_foo, __wrap___wrap_foo
foo:
  ret
__wrap_foo:
  ret
__wrap___wrap_foo:
  ret

_start:
  call foo
  call __wrap_foo
  movl $sym_foo, %eax
