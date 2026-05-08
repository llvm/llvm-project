# REQUIRES: riscv
# RUN: rm -rf %t && mkdir %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=-relax %s -o 32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=-relax %s -o 64.o

# RUN: ld.lld 32.o --defsym foo=_start+12 --defsym bar=_start -o out.32
# RUN: ld.lld 64.o --defsym foo=_start+12 --defsym bar=_start -o out.64
# RUN: llvm-objdump -d --no-show-raw-insn out.32 | FileCheck %s
# RUN: llvm-objdump -d --no-show-raw-insn out.64 | FileCheck %s
# RUN: ld.lld -pie 32.o --defsym foo=_start+12 --defsym bar=_start -o out.32
# RUN: ld.lld -pie 64.o --defsym foo=_start+12 --defsym bar=_start -o out.64
# RUN: llvm-objdump -d --no-show-raw-insn out.32 | FileCheck %s
# RUN: llvm-objdump -d --no-show-raw-insn out.64 | FileCheck %s
# CHECK:      auipc   a0, 0x0
# CHECK-NEXT: addi    a0, a0, 0xc
# CHECK-NEXT: sw      zero, 0xc(a0)
# CHECK:      auipc   a0, 0x0
# CHECK-NEXT: addi    a0, a0, -0xc
# CHECK-NEXT: sw      zero, -0xc(a0)

# RUN: ld.lld 32.o --defsym foo=_start+0x7ffff7ff --defsym bar=_start+12-0x80000800 -o out.32.limits
# RUN: ld.lld 64.o --defsym foo=_start+0x7ffff7ff --defsym bar=_start+12-0x80000800 -o out.64.limits
# RUN: llvm-objdump -d --no-show-raw-insn out.32.limits | FileCheck --check-prefix=LIMITS %s
# RUN: llvm-objdump -d --no-show-raw-insn out.64.limits | FileCheck --check-prefix=LIMITS %s
# LIMITS:      auipc   a0, 0x7ffff
# LIMITS-NEXT: addi    a0, a0, 0x7ff
# LIMITS-NEXT: sw      zero, 0x7ff(a0)
# LIMITS:      auipc   a0, 0x80000
# LIMITS-NEXT: addi    a0, a0, -0x800
# LIMITS-NEXT: sw      zero, -0x800(a0)

## rv32 handles R_RISCV_PCREL_HI20 with 32-bit overflow; rv64 produces an error.
# RUN: ld.lld 32.o --defsym foo=_start+0x7ffff800 --defsym bar=_start+12-0x80000801 -o /dev/null
# RUN: not ld.lld 64.o --defsym foo=_start+0x7ffff800 --defsym bar=_start+12-0x80000801 2>&1 | FileCheck --check-prefix=ERROR %s --implicit-check-not=error:
# ERROR: error: {{.*}}:(.text+0x0): relocation R_RISCV_PCREL_HI20 out of range: 524288 is not in [-524288, 524287]; references 'foo'
# ERROR: error: {{.*}}:(.text+0xc): relocation R_RISCV_PCREL_HI20 out of range: -524289 is not in [-524288, 524287]; references 'bar'

.global _start
_start:
    auipc   a0, %pcrel_hi(foo)
    addi    a0, a0, %pcrel_lo(_start)
    sw      x0, %pcrel_lo(_start)(a0)
.L1:
    auipc   a0, %pcrel_hi(bar)
    addi    a0, a0, %pcrel_lo(.L1)
    sw      x0, %pcrel_lo(.L1)(a0)
