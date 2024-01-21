# REQUIRES: riscv

# RUN: llvm-mc -filetype=obj -triple=riscv32-unknown-elf -mattr=-relax %s -o %t.rv32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64-unknown-elf -mattr=-relax %s -o %t.rv64.o

# RUN: ld.lld %t.rv32.o --defsym foo=_start+8 --defsym bar=_start -o %t.rv32
# RUN: ld.lld %t.rv64.o --defsym foo=_start+8 --defsym bar=_start -o %t.rv64
# RUN: llvm-objdump -d %t.rv32 | FileCheck %s
# RUN: llvm-objdump -d %t.rv64 | FileCheck %s
# CHECK:      97 00 00 00     auipc   ra, 0x0
# CHECK-NEXT: e7 80 80 00     jalr    0x8(ra)
# CHECK:      97 00 00 00     auipc   ra, 0x0
# CHECK-NEXT: e7 80 80 ff     jalr    -0x8(ra)

# RUN: ld.lld %t.rv32.o --defsym foo=_start+0x7ffff7ff --defsym bar=_start+8-0x80000800 -o %t.rv32.limits
# RUN: ld.lld %t.rv64.o --defsym foo=_start+0x7ffff7ff --defsym bar=_start+8-0x80000800 -o %t.rv64.limits
# RUN: llvm-objdump -d %t.rv32.limits | FileCheck --check-prefix=LIMITS %s
# RUN: llvm-objdump -d %t.rv64.limits | FileCheck --check-prefix=LIMITS %s
# LIMITS:      97 f0 ff 7f     auipc   ra, 0x7ffff
# LIMITS-NEXT: e7 80 f0 7f     jalr    0x7ff(ra)
# LIMITS-NEXT: 97 00 00 80     auipc   ra, 0x80000
# LIMITS-NEXT: e7 80 00 80     jalr    -0x800(ra)

# RUN: ld.lld %t.rv32.o --defsym foo=_start+0x7ffff800 --defsym bar=_start+8-0x80000801 -o %t
# RUN: not ld.lld %t.rv64.o --defsym foo=_start+0x7ffff800 --defsym bar=_start+8-0x80000801 -o /dev/null 2>&1 | \
# RUN:   FileCheck --check-prefix=ERROR %s
# ERROR: relocation R_RISCV_CALL_PLT out of range: 524288 is not in [-524288, 524287]; references 'foo'
# ERROR: relocation R_RISCV_CALL_PLT out of range: -524289 is not in [-524288, 524287]; references 'bar'

.global _start
_start:
    call    foo
    call    bar
