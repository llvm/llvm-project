# REQUIRES: riscv
# RUN: rm -rf %t && mkdir %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=-relax %s -o 32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=-relax %s -o 64.o

# RUN: ld.lld 32.o --defsym foo=_start+4 --defsym bar=_start -o out.32
# RUN: ld.lld 64.o --defsym foo=_start+4 --defsym bar=_start -o out.64
# RUN: llvm-objdump -d out.32 | FileCheck %s --check-prefix=CHECK-32
# RUN: llvm-objdump -d out.64 | FileCheck %s --check-prefix=CHECK-64
# CHECK-32: 0040006f    j   0x110b8
# CHECK-32: ffdff0ef    jal 0x110b4
# CHECK-64: 0040006f    j   0x11124
# CHECK-64: ffdff0ef    jal 0x11120

# RUN: ld.lld 32.o --defsym foo=_start+0xffffe --defsym bar=_start+4-0x100000 -o out.32.limits
# RUN: ld.lld 64.o --defsym foo=_start+0xffffe --defsym bar=_start+4-0x100000 -o out.64.limits
# RUN: llvm-objdump -d out.32.limits | FileCheck --check-prefix=LIMITS-32 %s
# RUN: llvm-objdump -d out.64.limits | FileCheck --check-prefix=LIMITS-64 %s
# LIMITS-32:      7ffff06f j   0x1110b2
# LIMITS-32-NEXT: 800000ef jal 0xfff110b8
# LIMITS-64:      7ffff06f j   0x11111e
# LIMITS-64-NEXT: 800000ef jal 0xfffffffffff11124

# RUN: not ld.lld 32.o --defsym foo=_start+0x100000 --defsym bar=_start+4-0x100002 2>&1 | FileCheck --check-prefix=ERROR-RANGE %s --implicit-check-not=error:
# RUN: not ld.lld 64.o --defsym foo=_start+0x100000 --defsym bar=_start+4-0x100002 2>&1 | FileCheck --check-prefix=ERROR-RANGE %s --implicit-check-not=error:
# ERROR-RANGE: error: {{.*}}:(.text+0x0): relocation R_RISCV_JAL out of range: 1048576 is not in [-1048576, 1048575]; references 'foo'
# ERROR-RANGE: error: {{.*}}:(.text+0x4): relocation R_RISCV_JAL out of range: -1048578 is not in [-1048576, 1048575]; references 'bar'

# RUN: not ld.lld 32.o --defsym foo=_start+1 --defsym bar=_start+4+3 2>&1 | FileCheck --check-prefix=ERROR-ALIGN %s --implicit-check-not=error:
# RUN: not ld.lld 64.o --defsym foo=_start+1 --defsym bar=_start+4+3 2>&1 | FileCheck --check-prefix=ERROR-ALIGN %s --implicit-check-not=error:
# ERROR-ALIGN: error: {{.*}}:(.text+0x0): improper alignment for relocation R_RISCV_JAL: 0x1 is not aligned to 2 bytes
# ERROR-ALIGN: error: {{.*}}:(.text+0x4): improper alignment for relocation R_RISCV_JAL: 0x3 is not aligned to 2 bytes

.global _start

_start:
    jal x0, foo
    jal x1, bar
