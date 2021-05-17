# REQUIRES: x86

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/exec.s -o %texec.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/shared.s -o %tshared.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/ledef.s -o %tdef.o
# RUN: ld.lld %texec.o -o %t.exec
# RUN: ld.lld %tshared.o -o %t.shared --shared
# RUN: llvm-objdump -d %t.exec | FileCheck %s --check-prefix=EXEC
# RUN: llvm-objdump -d %t.shared | FileCheck %s --check-prefix=SHARED

## An undefined weak TLS symbol does not fetch a lazy definition.
# RUN: ld.lld %texec.o --start-lib %tdef.o --end-lib -o %tlazy
# RUN: llvm-objdump -d %tlazy | FileCheck %s --check-prefix=EXEC

## Undefined TLS symbols arbitrarily resolve to 0.
# EXEC:   leaq 16(%rax), %rdx
## Initial-exec references to undefined weak symbols can be relaxed to LE
## references.
# EXEC:   leaq 32(%rax), %rax
# SHARED: leaq 48(%rax), %rcx

# RUN: ld.lld -shared %tdef.o -o %tdef.so
# RUN: not ld.lld %texec.o %tdef.so -o /dev/null 2>&1 | FileCheck --check-prefix=ERROR %s

# ERROR: symbol 'le' has no type

#--- ledef.s
.tbss
.globl le
le:

#--- exec.s
.weak le
leaq le@tpoff+16(%rax), %rdx

.weak ie
addq ie@gottpoff+32(%rip), %rax

#--- shared.s
.weak ld
leaq ld@dtpoff+48(%rax), %rcx
