# REQUIRES: ppc
# RUN: llvm-mc -filetype=obj -triple=powerpc %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefix=PDE %s
# RUN: ld.lld -pie %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefix=PIC %s
# RUN: ld.lld -shared %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefix=PIC %s

## It does not really matter how we fixup it, but we cannot overflow and
## should not generate a call stub (this would waste space).
# PDE: bl 0x100100b4

## With -pie or -shared, create a call stub. ld.bfd produces bl .+0
# PIC:       bl 0x[[PLT:[0-9a-f]+]]
# PIC-EMPTY:
# PIC-NEXT:  000[[PLT]] <00000000.plt_pic32.foo>:

.weak foo
bl foo
