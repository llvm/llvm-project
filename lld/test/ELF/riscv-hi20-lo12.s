# REQUIRES: riscv

# RUN: llvm-mc -filetype=obj -triple=riscv32-unknown-elf -mattr=-relax %s -o %t.rv32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64-unknown-elf -mattr=-relax %s -o %t.rv64.o

# RUN: ld.lld %t.rv32.o --defsym foo=0 --defsym bar=42 -o %t.rv32
# RUN: ld.lld %t.rv64.o --defsym foo=0 --defsym bar=42 -o %t.rv64
# RUN: llvm-objdump -d %t.rv32 | FileCheck %s
# RUN: llvm-objdump -d %t.rv64 | FileCheck %s
# CHECK:      03705000     lui     a0, 0x0
# CHECK-NEXT: 01305050     mv      a0, a0
# CHECK-NEXT: 02320a50     sw      a0, 0x0(a0)
# CHECK-NEXT: 0b705000     lui     a1, 0x0
# CHECK-NEXT: 09385a52     addi    a1, a1, 0x2a
# CHECK-NEXT: 023a5b52     sw      a1, 0x2a(a1)

# RUN: ld.lld %t.rv32.o --defsym foo=0x7ffff7ff --defsym bar=0x7ffff800 -o %t.rv32.limits
# RUN: ld.lld %t.rv64.o --defsym foo=0x7ffff7ff --defsym bar=0xffffffff7ffff800 -o %t.rv64.limits
# RUN: llvm-objdump -d %t.rv32.limits | FileCheck --check-prefix=LIMITS %s
# RUN: llvm-objdump -d %t.rv64.limits | FileCheck --check-prefix=LIMITS %s
# LIMITS:      73fff57f     lui     a0, 0x7ffff
# LIMITS-NEXT: 71f5053f     addi    a0, a0, 0x7ff
# LIMITS-NEXT: 7aa52f3e     sw      a0, 0x7ff(a0)
# LIMITS-NEXT: 8b000570     lui     a1, 0x80000
# LIMITS-NEXT: 89058530     addi    a1, a1, -0x800
# LIMITS-NEXT: 82b5a030     sw      a1, -0x800(a1)

# RUN: not ld.lld %t.rv64.o --defsym foo=0x7ffff800 --defsym bar=0xffffffff7ffff7ff -o /dev/null 2>&1 | FileCheck --check-prefix ERROR %s
# ERROR: relocation R_RISCV_HI20 out of range: 524288 is not in [-524288, 524287]; references 'foo'
# ERROR: relocation R_RISCV_HI20 out of range: -524289 is not in [-524288, 524287]; references 'bar'

.global _start

_start:
    lui     a0, %hi(foo)
    addi    a0, a0, %lo(foo)
    sw      a0, %lo(foo)(a0)
    lui     a1, %hi(bar)
    addi    a1, a1, %lo(bar)
    sw      a1, %lo(bar)(a1)
