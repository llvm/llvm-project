# REQUIRES: riscv

## When there are too few calls, table jump relaxation should not be profitable.
## Verify the .riscv.jvt section has zero size and no cm.jt/cm.jalt are emitted.

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax,+zcmt %s -o %t.rv32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax,+zcmt %s -o %t.rv64.o
# RUN: ld.lld %t.rv32.o --relax-tbljal --defsym foo=0x150000 -o %t.rv32
# RUN: ld.lld %t.rv64.o --relax-tbljal --defsym foo=0x150000 -o %t.rv64
# RUN: llvm-readelf -S %t.rv32 | FileCheck --check-prefix=SEC32 %s
# RUN: llvm-readelf -S %t.rv64 | FileCheck --check-prefix=SEC64 %s
# RUN: llvm-objdump -d --mattr=+zcmt --no-show-raw-insn %t.rv32 | FileCheck --check-prefix=DISASM %s
# RUN: llvm-objdump -d --mattr=+zcmt --no-show-raw-insn %t.rv64 | FileCheck --check-prefix=DISASM %s

# SEC32: .riscv.jvt PROGBITS {{[0-9a-f]+}} {{[0-9a-f]+}} 000000
# SEC64: .riscv.jvt PROGBITS {{[0-9a-f]+}} {{[0-9a-f]+}} 000000

# DISASM-NOT: cm.jt
# DISASM-NOT: cm.jalt

.global _start
.p2align 3
_start:
  call foo
  tail foo_1
  tail foo_2
  tail foo_3

foo_1:
  nop
foo_2:
  nop
foo_3:
  nop
