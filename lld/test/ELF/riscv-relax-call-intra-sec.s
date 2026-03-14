# REQUIRES: riscv
## Test R_RISCV_CALL referencing the current input section with the displacement
## close to the boundary.

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax %s -o %t.o
# RUN: ld.lld -Ttext=0x10000 %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases %t | FileCheck %s

# CHECK-LABEL:  <_start>:
# CHECK-NEXT:             jal    ra, {{.*}} <_start>
# CHECK-NEXT:             jal    ra, {{.*}} <_start>
# CHECK-EMPTY:
# CHECK-NEXT:   <a>:
# CHECK-NEXT:             c.jr   ra

# CHECK-LABEL:  <b>:
# CHECK:                  jal    zero, {{.*}} <a>
# CHECK-NEXT:             jal    zero, {{.*}} <c>
# CHECK-NEXT:             c.j    {{.*}} <c>

# CHECK-LABEL:  <c>:
# CHECK-NEXT:             c.jr   ra

#--- a.s
.global _start
_start:
  call _start
  call _start

a:
  ret
b:
  .space 2048
## Relaxed to jal. If we don't compute the precise value of a, we may consider
## a reachable by c.j.
  tail a
## Relaxed to jal. c.j is unreachable.
  tail c      # c.j
## Relaxed to c.j. If we don't compute the precise value of c, we may consider
## c.j unreachable.
  tail c      # c.j
  .space 2042
c:
  ret
