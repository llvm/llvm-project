## Exercise long jumps from fragments beyond the first cold fragment, and
## two predecessors sharing the same target and scratch register. Padding every
## fragment keeps all four trampolines outside the JAL range.

# RUN: llvm-mc -triple riscv64 -filetype=obj %s -o %t.o
# RUN: ld.lld --emit-relocs -e _start %t.o -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt --split-functions --split-strategy=all \
# RUN:   --pad-funcs-before=_start:2097152
# RUN: llvm-objdump -d \
# RUN:   --disassemble-symbols=_start,_start.cold.0,_start.cold.1,_start.cold.2 %t.bolt | FileCheck %s

# CHECK-LABEL: <_start>:
# CHECK: auipc t0,
# CHECK-NEXT: jr {{.*}} <_start.cold.0>
# CHECK: auipc t0,
# CHECK-NEXT: jr {{.*}} <_start.cold.1>
# CHECK-LABEL: <_start.cold.0>:
# CHECK-NEXT: addi a0, a0, 0x1
# CHECK-NEXT: auipc t0,
# CHECK-NEXT: jr {{.*}} <_start.cold.2>
# CHECK-LABEL: <_start.cold.1>:
# CHECK-NEXT: addi a0, a0, 0x2
# CHECK-NEXT: auipc t0,
# CHECK-NEXT: jr {{.*}} <_start.cold.2>
# CHECK-LABEL: <_start.cold.2>:
# CHECK-NEXT: li t0, 0x0
# CHECK-NEXT: ret

  .text
  .globl _start
  .type _start, @function
_start:
  .cfi_startproc
  beqz a0, .Lright
.Lleft:
  addi a0, a0, 1
  j .Ljoin
.Lright:
  addi a0, a0, 2
.Ljoin:
  li t0, 0
  ret
  .cfi_endproc
  .size _start, .-_start
  .reloc 0, R_RISCV_NONE
