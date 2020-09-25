# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o

# RUN: %lld %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s
# CHECK:      leaq {{.*}} # 100000000
# CHECK-NEXT: leaq {{.*}} # 100000000

# RUN: %lld -dylib %t.o -o %t.dylib
# RUN: llvm-objdump -d --no-show-raw-insn %t.dylib | FileCheck %s --check-prefix=DYLIB-CHECK
# DYLIB-CHECK:      leaq {{.*}} # 0
# DYLIB-CHECK-NEXT: leaq {{.*}} # 0

.globl _main
.text
_main:
  leaq ___dso_handle(%rip), %rdx
  movq ___dso_handle@GOTPCREL(%rip), %rdx
  ret
