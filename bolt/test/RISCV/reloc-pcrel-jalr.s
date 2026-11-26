## Check that BOLT preserves PC-relative operands on JALR calls and jumps
## and recomputes their displacements after moving the targets.

# RUN: llvm-mc -triple=riscv64 -filetype=obj %s -o %t.o
# RUN: ld.lld --emit-relocs %t.o -o %t.exe
# RUN: llvm-bolt %t.exe --print-disasm --print-only=_start,jump -o %t.bolt \
# RUN:   | FileCheck %s
# RUN: llvm-objdump -d -M no-aliases -j .text %t.bolt \
# RUN:   | FileCheck %s --check-prefix=REWRITE

# CHECK-LABEL: Binary Function "_start" after disassembly {
# CHECK:       auipc t0, %pcrel_hi(callee) # Label: [[CALL_HI:\.Ltmp[0-9]+]]
# CHECK-NEXT:  jalr t0, %pcrel_lo([[CALL_HI]])(t0)

# CHECK-LABEL: Binary Function "jump" after disassembly {
# CHECK:       auipc t1, %pcrel_hi(_start) # Label: [[JUMP_HI:\.Ltmp[0-9]+]]
# CHECK-NEXT:  jalr zero, %pcrel_lo([[JUMP_HI]])(t1)

# REWRITE-LABEL: <_start>:
# REWRITE:       auipc t0,
# REWRITE-NEXT:  jalr t0, {{.*}}(t0) <callee>
# REWRITE-LABEL: <jump>:
# REWRITE:       auipc t1,
# REWRITE-NEXT:  jalr zero, {{.*}}(t1) <_start>

  .text
  .globl _start
  .type _start, @function
_start:
.Lcall_hi:
  auipc t0, %pcrel_hi(callee)
  jalr t0, %pcrel_lo(.Lcall_hi)(t0)
  ret
  .size _start, .-_start

  .globl callee
  .type callee, @function
callee:
  jr t0
  .size callee, .-callee

  .globl jump
  .type jump, @function
jump:
.Ljump_hi:
  auipc t1, %pcrel_hi(_start)
  jalr zero, %pcrel_lo(.Ljump_hi)(t1)
  .size jump, .-jump
