# REQUIRES: riscv
# RUN: rm -rf %t && mkdir %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=-relax %s -o 32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=-relax %s -o 64.o

# RUN: ld.lld 32.o --defsym foo=0 --defsym bar=42 -o out.32
# RUN: ld.lld 64.o --defsym foo=0 --defsym bar=42 -o out.64
# RUN: llvm-objdump -d out.32 | FileCheck %s
# RUN: llvm-objdump -d out.64 | FileCheck %s
# CHECK:      00000537     lui     a0, 0x0
# CHECK-NEXT: 00050513     mv      a0, a0
# CHECK-NEXT: 00a52023     sw      a0, 0x0(a0)
# CHECK-NEXT: 000005b7     lui     a1, 0x0
# CHECK-NEXT: 02a58593     addi    a1, a1, 0x2a
# CHECK-NEXT: 02b5a523     sw      a1, 0x2a(a1)

# RUN: ld.lld 32.o --defsym foo=0x7ffff7ff --defsym bar=0x7ffff800 -o out.32.limits
# RUN: ld.lld 64.o --defsym foo=0x7ffff7ff --defsym bar=0xffffffff7ffff800 -o out.64.limits
# RUN: llvm-objdump -d out.32.limits | FileCheck --check-prefix=LIMITS %s
# RUN: llvm-objdump -d out.64.limits | FileCheck --check-prefix=LIMITS %s
# LIMITS:      7ffff537     lui     a0, 0x7ffff
# LIMITS-NEXT: 7ff50513     addi    a0, a0, 0x7ff
# LIMITS-NEXT: 7ea52fa3     sw      a0, 0x7ff(a0)
# LIMITS-NEXT: 800005b7     lui     a1, 0x80000
# LIMITS-NEXT: 80058593     addi    a1, a1, -0x800
# LIMITS-NEXT: 80b5a023     sw      a1, -0x800(a1)

# RUN: not ld.lld 64.o --defsym foo=0x7ffff800 --defsym bar=0xffffffff7ffff7ff 2>&1 | FileCheck --check-prefix=ERROR %s --implicit-check-not=error:
# ERROR: error: {{.*}}:(.text+0x0): relocation R_RISCV_HI20 out of range: 524288 is not in [-524288, 524287]; references 'foo'
# ERROR: error: {{.*}}:(.text+0xc): relocation R_RISCV_HI20 out of range: -524289 is not in [-524288, 524287]; references 'bar'

.global _start

_start:
    lui     a0, %hi(foo)
    addi    a0, a0, %lo(foo)
    sw      a0, %lo(foo)(a0)
    lui     a1, %hi(bar)
    addi    a1, a1, %lo(bar)
    sw      a1, %lo(bar)(a1)
