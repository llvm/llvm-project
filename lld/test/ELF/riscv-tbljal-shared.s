# REQUIRES: riscv

## Zcmt cannot be used with position-independent binaries (per the RISCV psABI).
## Verify that --relax-tbljal is ignored when linking shared-objects or
## dynamic executables.

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax,+zcmt %s -o %t.o

## Case 1: the output is a shared object.
# RUN: ld.lld -shared %t.o --relax-tbljal --defsym far=0x200000 -o %t.so
# RUN: llvm-readelf -S %t.so | FileCheck --check-prefix=NO-JVT %s
# RUN: llvm-objdump -d --mattr=+zcmt --no-show-raw-insn %t.so | FileCheck --check-prefix=NO-CM %s

## Case 2: linking against a shared object.
# RUN: echo '.globl bar; bar: ret' | llvm-mc -filetype=obj -triple=riscv32 -o %t-lib.o
# RUN: ld.lld -shared %t-lib.o -o %t-lib.so
# RUN: ld.lld %t.o %t-lib.so --relax-tbljal --defsym far=0x200000 -o %t.dyn
# RUN: llvm-readelf -S %t.dyn | FileCheck --check-prefix=NO-JVT %s
# RUN: llvm-objdump -d --mattr=+zcmt --no-show-raw-insn %t.dyn | FileCheck --check-prefix=NO-CM %s

# NO-JVT-NOT: .riscv.jvt
# NO-CM-NOT:  cm.jt
# NO-CM-NOT:  cm.jalt

.global _start
.p2align 3
_start:
  .rept 20
  tail far
  .endr
