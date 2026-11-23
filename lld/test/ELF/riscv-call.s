# REQUIRES: riscv
# RUN: rm -rf %t && mkdir %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=-relax %s -o 32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=-relax %s -o 64.o

# RUN: ld.lld 32.o --defsym foo=_start+8 --defsym bar=_start -o out.32
# RUN: ld.lld 64.o --defsym foo=_start+8 --defsym bar=_start -o out.64
# RUN: llvm-objdump -d out.32 | FileCheck %s
# RUN: llvm-objdump -d out.64 | FileCheck %s
# CHECK:      00000097     auipc   ra, 0x0
# CHECK-NEXT: 008080e7     jalr    0x8(ra)
# CHECK:      00000097     auipc   ra, 0x0
# CHECK-NEXT: ff8080e7     jalr    -0x8(ra)

# RUN: ld.lld 32.o --defsym foo=_start+0x7ffff7ff --defsym bar=_start+8-0x80000800 -o out.32.limits
# RUN: ld.lld 64.o --defsym foo=_start+0x7ffff7ff --defsym bar=_start+8-0x80000800 -o out.64.limits
# RUN: llvm-objdump -d out.32.limits | FileCheck --check-prefix=LIMITS %s
# RUN: llvm-objdump -d out.64.limits | FileCheck --check-prefix=LIMITS %s
# LIMITS:      7ffff097     auipc   ra, 0x7ffff
# LIMITS-NEXT: 7ff080e7     jalr    0x7ff(ra)
# LIMITS-NEXT: 80000097     auipc   ra, 0x80000
# LIMITS-NEXT: 800080e7     jalr    -0x800(ra)

## rv32 handles R_RISCV_CALL_PLT with 32-bit overflow; rv64 produces an error.
# RUN: ld.lld 32.o --defsym foo=_start+0x7ffff800 --defsym bar=_start+8-0x80000801 -o /dev/null
# RUN: not ld.lld 64.o --defsym foo=_start+0x7ffff800 --defsym bar=_start+8-0x80000801 2>&1 | FileCheck --check-prefix=ERROR %s --implicit-check-not=error:
# ERROR: error: {{.*}}:(.text+0x0): relocation R_RISCV_CALL_PLT out of range: 524288 is not in [-524288, 524287]; references 'foo'
# ERROR: error: {{.*}}:(.text+0x8): relocation R_RISCV_CALL_PLT out of range: -524289 is not in [-524288, 524287]; references 'bar'

.global _start
_start:
    call    foo
    call    bar
