# REQUIRES: riscv
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax a.s -o a.o

# RUN: ld.lld -T lds a.o -o a
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases a | FileCheck %s

## Unsure whether this needs a diagnostic. GNU ld allows this.
# RUN: ld.lld -T lds -pie a.o -o a.pie
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases a.pie | FileCheck %s

# RUN: ld.lld -T lds -pie -z notext -z ifunc-noplt a.o -o a.ifunc-noplt
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases a.ifunc-noplt | FileCheck %s --check-prefix=CHECK2

# CHECK-LABEL:  <_start>:
# CHECK-NEXT:             jal    zero, 0x8 <abs>
# CHECK-NEXT:             jal    zero, 0x8 <abs>
# CHECK-NEXT:             jal    ra, 0x8 <abs>
# CHECK-NEXT:             auipc  t1, 1048320
# CHECK-NEXT:             jalr   zero, -4(t1)
# CHECK-EMPTY:

# CHECK-LABEL:  <.mid>:
# CHECK-NEXT:             jal    zero, 0x101000
# CHECK-NEXT:             c.j    0x101000
# CHECK-EMPTY:

# CHECK2-LABEL: <.mid>:
# CHECK2-NEXT:            auipc  t1, 0
# CHECK2-NEXT:            jalr   zero, 0(t1)
# CHECK2-NEXT:            auipc  t1, 0
# CHECK2-NEXT:            jalr   zero, 0(t1)
# CHECK2-EMPTY:

#--- a.s
.global _start, ifunc
_start:
  jump abs, t2
  jump abs, t3
  call abs
  tail abs       # not relaxed

.section .mid,"ax",@progbits
.balign 16
  tail ifunc@plt # not relaxed
  tail ifunc@plt

.type ifunc, @gnu_indirect_function
ifunc:
  ret

#--- lds
SECTIONS {
  .text 0x100000 : { *(.text) }
  .mid 0x100800 : { *(.mid) }
  .iplt 0x101000 : { *(.iplt) }
}
abs = 8;
