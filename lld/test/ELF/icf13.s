# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1
# RUN: ld.lld -shared -z notext %t1 -o %t --icf=all --print-icf-sections 2>&1 | FileCheck -allow-empty %s

## Check ICF does not collect sections which relocations point to symbols
## of the different types. Like to defined and undefined symbols in this test case.

# CHECK-NOT: selected

.globl und

.section .text
.globl _start
_start:
  ret

.section .text.foo, "ax"
.quad _start

.section .text.bar, "ax"
.quad und
