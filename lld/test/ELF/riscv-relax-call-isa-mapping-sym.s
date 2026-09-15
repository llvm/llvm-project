# REQUIRES: riscv
## R_RISCV_RELAX's ISA mapping symbol ($x<isa-string>) drives per-region RVC
## relaxation, overriding the file-level EF_RISCV_RVC flag. Built with +c,+relax,
## so regions whose mapping symbol lacks C must still avoid c.j.

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax %s -o %t.o
# RUN: llvm-readobj -r %t.o | FileCheck %s --check-prefix=RELOC
# RUN: ld.lld %t.o -Ttext=0x10000 -o %t
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases %t | FileCheck %s

## No .option arch yet: initial mapping symbol includes C (from +c) -> c.j.
# RELOC: R_RISCV_RELAX $xrv64{{.*}}_c2p0
# CHECK-LABEL: <compat>:
# CHECK-NEXT: {{.*}}: c.j {{.*}} <target>
.globl compat
compat:
    tail target

## C in mapping symbol -> c.j.
# CHECK-LABEL: <with_c>:
# CHECK-NEXT: {{.*}}: c.j {{.*}} <target>
.option arch, rv64imafdc
.globl with_c
with_c:
    tail target

## No C in mapping symbol -> jal, despite file-level EF_RISCV_RVC.
# CHECK-LABEL: <without_c>:
# CHECK-NEXT: {{.*}}: jal zero, {{.*}} <target>
.option arch, rv64imafd
.globl without_c
without_c:
    tail target

## .option norvc drops C -> jal.
# CHECK-LABEL: <norvc_region>:
# CHECK-NEXT: {{.*}}: jal zero, {{.*}} <target>
.option arch, rv64imafdc
.option norvc
.globl norvc_region
norvc_region:
    tail target

## .option rvc restores C -> c.j.
# CHECK-LABEL: <after_rvc>:
# CHECK-NEXT: {{.*}}: c.j {{.*}} <target>
.option rvc
.globl after_rvc
after_rvc:
    tail target

.globl target
target:
    ret
