# REQUIRES: x86
# RUN: mkdir -p %t
# RUN: yaml2obj %p/Inputs/hello-dylib.yaml -o %t/libhello.dylib
# RUN: yaml2obj %p/Inputs/goodbye-dylib.yaml -o %t/libgoodbye.dylib
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/dylink.o
# RUN: lld -flavor darwinnew -o %t/dylink -Z -L%t -lhello -lgoodbye %t/dylink.o
# RUN: llvm-objdump --bind -d %t/dylink | FileCheck %s

# CHECK: movq [[#%u, HELLO_OFF:]](%rip), %rsi
# CHECK-NEXT: [[#%x, HELLO_RIP:]]:

# CHECK: movq [[#%u, GOODBYE_OFF:]](%rip), %rsi
# CHECK-NEXT: [[#%x, GOODBYE_RIP:]]:

# CHECK-LABEL: Bind table:
# CHECK-DAG: __DATA_CONST __got 0x{{0*}}[[#%x, HELLO_RIP + HELLO_OFF]]     pointer 0 libhello   _hello_world
# CHECK-DAG: __DATA_CONST __got 0x{{0*}}[[#%x, GOODBYE_RIP + GOODBYE_OFF]] pointer 0 libgoodbye _goodbye_world

.section __TEXT,__text
.globl _main

_main:
  movl $0x2000004, %eax # write() syscall
  mov $1, %rdi # stdout
  movq _hello_world@GOTPCREL(%rip), %rsi
  mov $13, %rdx # length of str
  syscall

  movl $0x2000004, %eax # write() syscall
  mov $1, %rdi # stdout
  movq _goodbye_world@GOTPCREL(%rip), %rsi
  mov $15, %rdx # length of str
  syscall
  mov $0, %rax
  ret
