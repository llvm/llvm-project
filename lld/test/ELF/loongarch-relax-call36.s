# REQUIRES: loongarch
## Relax R_LARCH_CALL36.
## Currently only loongarch64 is covered, because the call36 pseudo-instruction
## is valid for LA64 only, due to LA32 not having pcaddu18i.

# TODO:
#
# * trivial cases
# * +/- limit: -4, 0, +4
# * align: 0, 1, 2, 3
# * invalid pcaddu18i + jirl pairs
#    - rd1 != rj2
#    - rd2 not in (0, 1)

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=loongarch64 -mattr=+relax a.s -o a.o

# RUN: ld.lld -T lds a.o -o a
# RUN: llvm-objdump -d --no-show-raw-insn a | FileCheck %s

## Unsure whether this needs a diagnostic. GNU ld allows this.
# RUN: ld.lld -T lds -pie a.o -o a.pie
# RUN: llvm-objdump -d --no-show-raw-insn a.pie | FileCheck %s

# RUN: ld.lld -T lds -pie -z notext -z ifunc-noplt a.o -o a.ifunc-noplt
# RUN: llvm-objdump -d --no-show-raw-insn a.ifunc-noplt | FileCheck %s --check-prefix=CHECK2

# CHECK-LABEL:  <_start>:
# CHECK-NEXT:             bl -4 <near_before>
# CHECK-NEXT:             b -8 <near_before>
# CHECK-NEXT:             bl 64 <near_after>
# CHECK-NEXT:             b 60 <near_after>
# CHECK-NEXT:             pcaddu18i $ra, -512
# CHECK-NEXT:             jirl $ra, $ra, -4
# CHECK-NEXT:             bl -134217728 <far_b>
# CHECK-NEXT:             bl 134217724 <far_y>
# CHECK-NEXT:             pcaddu18i $ra, 512
# CHECK-NEXT:             jirl $ra, $ra, 0
# CHECK-NEXT:             pcaddu18i $t0, 0
# CHECK-NEXT:             jirl $t0, $t0, -44
# CHECK-NEXT:             pcaddu18i $t0, 0
# CHECK-NEXT:             jirl $zero, $t1, 24
# CHECK-NEXT:             pcalau12i $t0, 0
# CHECK-NEXT:             jirl $zero, $t0, -60
# CHECK-NEXT:             pcaddu18i $t0, 0
# CHECK-NEXT:             addu16i.d $t0, $t0, 2
# CHECK-EMPTY:

# CHECK-LABEL:  <.mid>:
# CHECK-NEXT:             b 2048
# CHECK-NEXT:             b 2044
# CHECK-EMPTY:

# CHECK2-LABEL: <.mid>:
# CHECK2-NEXT:            pcaddu18i $t0, 0
# CHECK2-NEXT:            jr $t0
# CHECK2-NEXT:            pcaddu18i $t0, 0
# CHECK2-NEXT:            jr $t0
# CHECK2-EMPTY:

#--- a.s
.global _start, ifunc
near_before:
  ret

_start:
  call36 near_before
  tail36 $t0, near_before

  call36 near_after
  tail36 $t0, near_after

  call36 far_a  ## just out of relaxable range: 0x08000010 - 0x10000014 = -(1 << 27) - 4
  call36 far_b  ## just in relaxable range: 0x0800001c - 0x1000001c = -(1 << 27)

  call36 far_y  ## just in relaxable range: 0x1800001c - 0x10000020 = (1 << 27) - 4
  call36 far_z  ## just out of relaxable range: 0x18000024 - 0x10000024 = 1 << 27

  ## broken R_LARCH_CALL36 usages should not be relaxed even if relaxable
  ## otherwise
  ## correctness is not guaranteed for malformed input like these

  ## jirl link register (rd) not $zero or $ra (hence not expressible by B or BL)
  ## the apparent correctness here is only coincidence and should not be relied
  ## upon
  .reloc ., R_LARCH_CALL36, near_before
  .reloc ., R_LARCH_RELAX, 0
  pcaddu18i $t0, 0
  jirl      $t0, $t0, 0

  ## jirl base != pcaddu18i output
  .reloc ., R_LARCH_CALL36, near_after
  .reloc ., R_LARCH_RELAX, 0
  pcaddu18i $t0, 0
  jirl      $zero, $t1, 0

  ## 1st insn not pcaddu18i
  .reloc ., R_LARCH_CALL36, near_before
  .reloc ., R_LARCH_RELAX, 0
  pcalau12i $t0, 0
  jirl      $zero, $t0, 0

  ## 2nd insn not jirl
  .reloc ., R_LARCH_CALL36, near_after
  .reloc ., R_LARCH_RELAX, 0
  pcaddu18i $t0, 0
  addu16i.d $t0, $t0, 0

near_after:
  ret

.section .mid,"ax",@progbits
.balign 16
  tail36 $t0, ifunc@plt
  tail36 $t0, ifunc@plt

.type ifunc, @gnu_indirect_function
ifunc:
  ret

#--- lds
SECTIONS {
  .text 0x10000000 : { *(.text) }
  .mid  0x10000800 : { *(.mid) }
  .iplt 0x10001000 : { *(.iplt) }
}

far_a = 0x08000010;
far_b = 0x0800001c;
far_y = 0x1800001c;
far_z = 0x18000024;
