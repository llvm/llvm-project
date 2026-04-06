# REQUIRES: riscv
## Tests that the .option norvc / .option rvc toggle drives per-region RVC
## relaxation through ISA mapping symbols.  The file has EF_RISCV_RVC set (+c),
## but the norvc region emits an ISA mapping symbol without C, so the linker
## must NOT use compressed jumps there.  After .option rvc, C is back and c.j
## is used again.

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax %s -o %t.o
# RUN: ld.lld %t.o -Ttext=0x10000 -o %t
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases %t | FileCheck %s

# CHECK-LABEL: <before_norvc>:
# CHECK-NEXT: {{.*}}: c.j {{.*}} <target>

# CHECK-LABEL: <norvc_region>:
# CHECK-NEXT: {{.*}}: jal zero, {{.*}} <target>

# CHECK-LABEL: <after_rvc>:
# CHECK-NEXT: {{.*}}: c.j {{.*}} <target>

## Toggle: start with RVC enabled (file has EF_RISCV_RVC via +c), disable via
## .option norvc (emits ISA mapping symbol without C), then re-enable via
## .option rvc (emits ISA mapping symbol with C).
.option arch, rv64imafdc
.globl before_norvc
before_norvc:
    tail target          ## has C mapping symbol → c.j

.option norvc
.globl norvc_region
norvc_region:
    tail target          ## no-C mapping symbol → jal zero (not c.j)

.option rvc
.globl after_rvc
after_rvc:
    tail target          ## C mapping symbol restored → c.j

.globl target
target:
    ret
