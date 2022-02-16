# REQUIRES: ppc

# RUN: llvm-mc -filetype=obj -triple=powerpc64le %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s --check-prefix=PDE
# RUN: ld.lld -pie %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s --check-prefix=PIC
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-objdump -d --no-show-raw-insn %t.so | FileCheck %s --check-prefix=PIC

# RUN: llvm-mc -filetype=obj -triple=powerpc64 %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s --check-prefix=PDE

## Branches to an undefined weak symbol need a thunk iff a dynamic relocation is
## produced. undefweak2 is hidden and does not need a dynamic relocation, so we
## suppress the thunk. undefweak1 needs a thunk iff -pie or -shared.

# PDE-LABEL: <_start>:
# PDE-NEXT:    bl {{.*}} <_start>
# PDE-NEXT:    nop
# PDE-NEXT:    bl {{.*}} <_start+0x8>
# PDE-NEXT:    nop

# PIC-LABEL: <_start>:
# PIC-NEXT:    bl {{.*}} <__plt_undefweak1>
# PIC-NEXT:    ld 2, 24(1)
# PIC-NEXT:    bl {{.*}} <_start+0x8>
# PIC-NEXT:    nop

.text
.global _start
_start:
  bl undefweak1
  nop
  bl undefweak2
  nop

.weak undefweak1, undefweak2
.hidden undefweak2
