# REQUIRES: riscv
# RUN: rm -rf %t && mkdir %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=riscv32 %s -o 32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 %s -o 64.o

# RUN: ld.lld 32.o --defsym foo=_start+4 --defsym bar=_start -o out.32
# RUN: ld.lld 64.o --defsym foo=_start+4 --defsym bar=_start -o out.64
# RUN: llvm-objdump -d out.32 | FileCheck %s --check-prefix=CHECK-32
# RUN: llvm-objdump -d out.64 | FileCheck %s --check-prefix=CHECK-64
# CHECK-32: 00000263     beqz    zero, 0x110b8
# CHECK-32: fe001ee3     bnez    zero, 0x110b4
# CHECK-64: 00000263     beqz    zero, 0x11124
# CHECK-64: fe001ee3     bnez    zero, 0x11120

# RUN: ld.lld 32.o --defsym foo=_start+0xffe --defsym bar=_start+4-0x1000 -o out.32.limits
# RUN: ld.lld 64.o --defsym foo=_start+0xffe --defsym bar=_start+4-0x1000 -o out.64.limits
# RUN: llvm-objdump -d out.32.limits | FileCheck --check-prefix=LIMITS-32 %s
# RUN: llvm-objdump -d out.64.limits | FileCheck --check-prefix=LIMITS-64 %s
# LIMITS-32:      7e000fe3     beqz    zero, 0x120b2
# LIMITS-32-NEXT: 80001063     bnez    zero, 0x100b8
# LIMITS-64:      7e000fe3     beqz    zero, 0x1211e
# LIMITS-64-NEXT: 80001063     bnez    zero, 0x10124

# RUN: not ld.lld 32.o --defsym foo=_start+0x1000 --defsym bar=_start+4-0x1002 2>&1 | FileCheck --check-prefix=ERROR-RANGE %s --implicit-check-not=error:
# RUN: not ld.lld 64.o --defsym foo=_start+0x1000 --defsym bar=_start+4-0x1002 2>&1 | FileCheck --check-prefix=ERROR-RANGE %s --implicit-check-not=error:
# ERROR-RANGE: error: {{.*}}:(.text+0x0): relocation R_RISCV_BRANCH out of range: 4096 is not in [-4096, 4095]; references 'foo'
# ERROR-RANGE: error: {{.*}}:(.text+0x4): relocation R_RISCV_BRANCH out of range: -4098 is not in [-4096, 4095]; references 'bar'

# RUN: not ld.lld 32.o --defsym foo=_start+1 --defsym bar=_start+4+1 2>&1 | FileCheck --check-prefix=ERROR-ALIGN %s --implicit-check-not=error:
# RUN: not ld.lld 64.o --defsym foo=_start+1 --defsym bar=_start+4+1 2>&1 | FileCheck --check-prefix=ERROR-ALIGN %s --implicit-check-not=error:
# ERROR-ALIGN: error: {{.*}}:(.text+0x0): improper alignment for relocation R_RISCV_BRANCH: 0x1 is not aligned to 2 bytes
# ERROR-ALIGN: error: {{.*}}:(.text+0x4): improper alignment for relocation R_RISCV_BRANCH: 0x1 is not aligned to 2 bytes

.option exact

.global _start
_start:
     beq x0, x0, foo
     bne x0, x0, bar
