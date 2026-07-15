## Check that BOLT recognizes a non-preemptible IFUNC IPLT entry and tracks
## the resolver when the linker canonicalizes the IFUNC symbol to the entry.

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax -o %t.o %s
# RUN: ld.lld -pie -q -o %t.exe %t.o
# RUN: llvm-bolt %t.exe -o %t.bolt --print-disasm --print-only=_start 2>&1 \
# RUN:   | FileCheck --check-prefix=BOLT %s
# RUN: llvm-readelf -r -s %t.bolt | FileCheck --check-prefix=ELF %s
## RV32 static binaries use a 32-bit wordclass IRELATIVE field. Also verify
## that a resolver at a secondary entry point retains its +4 offset.
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax -o %t.32.o %s
# RUN: ld.lld -q -o %t.32.exe %t.32.o
# RUN: llvm-bolt %t.32.exe -o %t.32.bolt
# RUN: llvm-readelf -r -s %t.32.bolt | FileCheck --check-prefix=RV32 %s

# BOLT: Binary Function "_start
# BOLT: auipc a0, %pcrel_hi(__BOLT_PSEUDO_.iplt)
# BOLT-NOT: unable to get new address corresponding to input address
# ELF: R_RISCV_IRELATIVE{{.*}}400044
# ELF: FUNC{{.*}}ifunc0
# RV32: R_RISCV_IRELATIVE{{.*}}400044
# RV32: FUNC{{.*}}ifunc0

  .text
  .globl _start
  .type _start, @function
_start:
1:
  auipc a0, %pcrel_hi(ifunc0)
  addi a0, a0, %pcrel_lo(1b)

  .globl func
  .type func, @function
func:
  ret

  .globl ifunc0
  .type ifunc0, @gnu_indirect_function
ifunc0:
  ret
