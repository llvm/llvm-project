# REQUIRES: riscv

# RUN: llvm-mc -filetype=obj -triple=riscv32-unknown-elf -mattr=-relax -riscv-asm-relax-branches=0 %s -o %t.rv32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64-unknown-elf -mattr=-relax -riscv-asm-relax-branches=0 %s -o %t.rv64.o

# RUN: ld.lld %t.rv32.o --defsym foo=_start+4 --defsym bar=_start -o %t.rv32
# RUN: ld.lld %t.rv64.o --defsym foo=_start+4 --defsym bar=_start -o %t.rv64
# RUN: llvm-objdump -d %t.rv32 | FileCheck %s --check-prefix=CHECK-32
# RUN: llvm-objdump -d %t.rv64 | FileCheck %s --check-prefix=CHECK-64
# CHECK-32: 00000263     beqz    zero, 0x110b8
# CHECK-32: fe001ee3     bnez    zero, 0x110b4
# CHECK-64: 00000263     beqz    zero, 0x11124
# CHECK-64: fe001ee3     bnez    zero, 0x11120
#
# RUN: ld.lld %t.rv32.o --defsym foo=_start+0xffe --defsym bar=_start+4-0x1000 -o %t.rv32.limits
# RUN: ld.lld %t.rv64.o --defsym foo=_start+0xffe --defsym bar=_start+4-0x1000 -o %t.rv64.limits
# RUN: llvm-objdump -d %t.rv32.limits | FileCheck --check-prefix=LIMITS-32 %s
# RUN: llvm-objdump -d %t.rv64.limits | FileCheck --check-prefix=LIMITS-64 %s
# LIMITS-32:      7e000fe3     beqz    zero, 0x120b2
# LIMITS-32-NEXT: 80001063     bnez    zero, 0x100b8
# LIMITS-64:      7e000fe3     beqz    zero, 0x1211e
# LIMITS-64-NEXT: 80001063     bnez    zero, 0x10124

# RUN: not ld.lld %t.rv32.o --defsym foo=_start+0x1000 --defsym bar=_start+4-0x1002 -o /dev/null 2>&1 | FileCheck --check-prefix=ERROR-RANGE %s
# RUN: not ld.lld %t.rv64.o --defsym foo=_start+0x1000 --defsym bar=_start+4-0x1002 -o /dev/null 2>&1 | FileCheck --check-prefix=ERROR-RANGE %s
# ERROR-RANGE: relocation R_RISCV_BRANCH out of range: 4096 is not in [-4096, 4095]; references 'foo'
# ERROR-RANGE: relocation R_RISCV_BRANCH out of range: -4098 is not in [-4096, 4095]; references 'bar'

# RUN: not ld.lld %t.rv32.o --defsym foo=_start+1 --defsym bar=_start-1 -o /dev/null 2>&1 | FileCheck --check-prefix=ERROR-ALIGN %s
# RUN: not ld.lld %t.rv64.o --defsym foo=_start+1 --defsym bar=_start-1 -o /dev/null 2>&1 | FileCheck --check-prefix=ERROR-ALIGN %s
# ERROR-ALIGN: improper alignment for relocation R_RISCV_BRANCH: 0x1 is not aligned to 2 bytes

.global _start
_start:
     beq x0, x0, foo
     bne x0, x0, bar
