# REQUIRES: riscv
## Tests that R_RISCV_RELAX ISA mapping symbols ($x<isa-string>, emitted by
## .option arch) control per-region RVC relaxation.
## Within a single object that has EF_RISCV_RVC set at the file level:
##   - a region with an ISA-with-C mapping symbol MUST use compressed jumps.
##   - a region with an ISA-without-C mapping symbol must NOT use compressed jumps,
##     even though EF_RISCV_RVC is set for the whole file.

## Build with +c,+relax so EF_RISCV_RVC is set at the file level.
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax %s -o %t.o
# RUN: ld.lld %t.o -Ttext=0x10000 -o %t
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases %t | FileCheck %s

## The call inside the rv64imafdc region should relax to compressed c.j.
# CHECK-LABEL: <with_c>:
# CHECK-NEXT: {{.*}}: c.j {{.*}} <target>

## The call inside the rv64imafd region must NOT use a compressed jump,
## even though EF_RISCV_RVC is set for the whole file.
# CHECK-LABEL: <without_c>:
# CHECK-NEXT: {{.*}}: jal zero, {{.*}} <target>

## Region 1: .option arch sets ISA to rv64imafdc (C extension present).
## The R_RISCV_RELAX for 'tail' will reference the $xrv64..._c2p0_... symbol.
.option arch, rv64imafdc
.globl with_c
with_c:
    tail target

## Region 2: .option arch removes the C extension.
## The R_RISCV_RELAX for 'tail' will reference the $xrv64..._zicsr... symbol
## (no c2p0), preventing compressed-jump relaxation.
.option arch, rv64imafd
.globl without_c
without_c:
    tail target

.globl target
target:
    ret
