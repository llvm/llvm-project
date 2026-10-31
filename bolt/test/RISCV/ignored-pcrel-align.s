## LLD's retained relocations can be inside their instructions after processing
## ALIGN with --no-relax. Resolve a low relocation's high instruction by range,
## so both halves of an address in retained code follow the moved target.
# RUN: llvm-mc -triple=riscv64 -mattr=+c,+relax -filetype=obj %s -o %t.o
# RUN: ld.lld --emit-relocs --no-relax -Ttext=0x10000 %t.o -o %t.exe
# RUN: llvm-bolt %t.exe --skip-funcs=_start --no-huge-pages --use-gnu-stack --align-text=4 --align-functions=4 --custom-allocation-vma=0x20000 -o %t.bolt
# RUN: llvm-objdump -d --no-show-raw-insn %t.bolt | FileCheck %s --check-prefix=CODE

## AUIPC is at 0x10008 and ADDI at 0x1000e, while their relocations are two
## bytes later. The low relocation still names the actual AUIPC address.
# CODE-LABEL: <.Lhi>:
# CODE-NEXT: auipc a0, 0x10
# CODE: addi a0, a0, -0x8
# CODE-LABEL: 0000000000020000 <target>:

  .text
  .globl _start, target
  .type _start, @function
_start:
  call target
  .p2align 2
.Lhi:
  auipc a0, %pcrel_hi(target)
  li a1, 7
  addi a0, a0, %pcrel_lo(.Lhi)
  ret
  .size _start, .-_start

  .type target, @function
target:
  li a0, 7
  ret
  .size target, .-target
