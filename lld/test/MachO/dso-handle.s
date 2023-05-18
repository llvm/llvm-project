# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o

# RUN: %lld -lSystem %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s
# CHECK:      leaq {{.*}} ## 0x100000000
# CHECK-NEXT: leaq {{.*}} ## 0x100000000

# RUN: %lld -dylib %t.o -o %t.dylib
# RUN: llvm-objdump -d --no-show-raw-insn --rebase --section-headers %t.dylib | FileCheck %s --check-prefix=DYLIB-CHECK
# DYLIB-CHECK:      leaq {{.*}} ## 0x0
# DYLIB-CHECK-NEXT: leaq {{.*}} ## 0x0

# DYLIB-LABEL: Sections:
# DYLIB:       __data        00000008 [[#%x,DATA:]] DATA
# DYLIB-LABEL: Rebase table:
# DYLIB-NEXT:  segment  section  address            type
# DYLIB-NEXT:  __DATA   __data   0x{{0*}}[[#DATA]]  pointer

# RUN: llvm-objdump --syms %t.dylib | FileCheck %s --check-prefix=SYMS
# SYMS-NOT: ___dso_handle

.globl _main
.text
_main:
  leaq ___dso_handle(%rip), %rdx
  movq ___dso_handle@GOTPCREL(%rip), %rdx
  ret

.data
.quad ___dso_handle
