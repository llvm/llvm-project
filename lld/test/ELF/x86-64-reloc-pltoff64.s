# REQUIRES: x86
## Test R_X86_64_PLTOFF64 (preemptible: L - GOT + A; non-preemptible: S - GOT + A).

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld %t.o -shared -o %t.so
# RUN: llvm-readelf -S %t.so | FileCheck %s --check-prefix=SEC-SHARED
# RUN: llvm-objdump -d --no-show-raw-insn %t.so | FileCheck %s --check-prefix=SHARED

# RUN: ld.lld %t.o -o %t
# RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=SEC-PDE
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s --check-prefix=PDE

# SEC-SHARED: .got.plt PROGBITS 00000000000033c0 0003c0 000028

## foo@plt - .got.plt = 0x12f0 - 0x33c0 = -8400
## undefweak@plt - .got.plt = 0x1300 - 0x33c0 = -8384
# SHARED-LABEL: <.text>:
# SHARED-NEXT:          movabsq $-8400, %rdx
# SHARED-NEXT:          movabsq $-8384, %rdx
# SHARED-LABEL: <foo@plt>:
# SHARED-NEXT:    12f0: jmpq {{.*}}(%rip)

# SEC-PDE: .got.plt PROGBITS 0000000000202170 000170 000018

## Avoid PLT since the referenced symbol is non-preemptible.
## foo - .got.plt = 0x20116c - 0x202170 = -4100
## 0 - .got.plt = 0 - 0x202168 = -2105712
# PDE-LABEL: <.text>:
# PDE-NEXT:            movabsq $-4100, %rdx
# PDE-NEXT:            movabsq $-2105712, %rdx
# PDE-LABEL: <foo>:
# PDE-NEXT:    20116c: retq

  movabsq $foo@PLTOFF, %rdx
  movabsq $undefweak@PLTOFF, %rdx

.globl foo
foo:
  ret

.weak undefweak
