# REQUIRES: riscv
## Tests that, even without a .option arch directive, R_RISCV_RELAX still
## references the initial "$x<InitialISA>" mapping symbol emitted before the
## first instruction.  For -mattr=+c,+relax that initial ISA includes the C
## extension, so the linker picks c.j via the ISA-mapping path (not the
## EF_RISCV_RVC fallback).

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax %s -o %t.o
# RUN: llvm-readobj -r %t.o | FileCheck %s --check-prefix=RELOC
# RUN: ld.lld %t.o -Ttext=0x10000 -o %t
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases %t | FileCheck %s

## Confirm that even without .option arch, the object gets a concrete ISA
## mapping symbol (not '-') on its R_RISCV_RELAX.
# RELOC: R_RISCV_RELAX $xrv64i2p1_c2p0_zca1p0

# CHECK-LABEL: <compat>:
# CHECK-NEXT: {{.*}}: c.j {{.*}} <compat_target>

## No .option arch directive.  R_RISCV_RELAX still references the initial
## "$x<InitialISA>" mapping symbol emitted before the first instruction, which
## for -mattr=+c,+relax includes the C extension; the linker therefore picks
## c.j via the ISA-mapping path (not the EF_RISCV_RVC fallback).
.globl compat
compat:
    tail compat_target

.globl compat_target
compat_target:
    ret
