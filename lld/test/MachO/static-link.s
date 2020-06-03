# REQUIRES: x86

# RUN: mkdir -p %t
#
# RUN: llvm-mc -filetype obj -triple x86_64-apple-darwin %p/Inputs/libgoodbye.s -o %t/goodbye.o
# RUN: llvm-ar --format=darwin crs %t/libgoodbye.a %t/goodbye.o
#
# RUN: llvm-mc -filetype obj -triple x86_64-apple-darwin %s -o %t/test.o
# RUN: lld -flavor darwinnew -o %t/test -Z -L%t -lgoodbye %t/test.o
#
# RUN: llvm-objdump --syms -d -r %t/test | FileCheck %s

# CHECK: SYMBOL TABLE:
# CHECK: {{0+}}[[ADDR:[0-9a-f]+]] g     O __TEXT,__cstring _goodbye_world

# CHECK: Disassembly of section __TEXT,__text
# CHECK-LABEL: <_main>:
# CHECK: leaq {{.*}}(%rip), %rsi   # [[ADDR]] <_goodbye_world>

.section __TEXT,__text
.global _main

_main:
  movl $0x2000004, %eax                 # write()
  mov $1, %rdi                          # stdout
  leaq _goodbye_world(%rip), %rsi
  mov $15, %rdx                         # length
  syscall
  mov $0, %rax
  ret
