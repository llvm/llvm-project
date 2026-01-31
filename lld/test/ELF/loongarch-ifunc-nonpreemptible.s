# REQUIRES: loongarch
# RUN: llvm-mc -filetype=obj -triple=loongarch64 -mattr=+relax %s -o %t.o
# RUN: ld.lld -pie %t.o -o %t
# RUN: llvm-readobj -r %t | FileCheck --check-prefix=RELOC %s
# RUN: llvm-readelf -s %t | FileCheck --check-prefix=SYM %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefix=DIS %s

## ifunc0 has a direct relocation, so it gets canonicalized to the IPLT entry.
## ifunc1 has only a GOT relocation, so its symbols remains in the original section.
## All IRELATIVE addends must be correctly adjusted after relaxation.

# RELOC:      .rela.dyn {
# RELOC-NEXT:   0x30380 R_LARCH_IRELATIVE - 0x10290
# RELOC-NEXT:   0x30388 R_LARCH_IRELATIVE - 0x10294
# RELOC-NEXT: }

# SYM:      {{0*}}102a0 0 FUNC  GLOBAL DEFAULT {{.*}} ifunc0
# SYM-NEXT: {{0*}}10294 0 IFUNC GLOBAL DEFAULT {{.*}} ifunc1

# DIS:      <_start>:
# DIS-NEXT:   10278: bl 20 <func>
# DIS-NEXT:          pcalau12i $a0, 0
# DIS-NEXT:          addi.d $a0, $a0, 672
# DIS-NEXT:          pcalau12i $a1, 32
# DIS-NEXT:          ld.d $a1, $a1, 904
# DIS:      Disassembly of section .iplt:
# DIS:      <ifunc0>:
# DIS-NEXT:   102a0: pcaddu12i $t3, 32

.text
.globl _start
_start:
  call36 func
.L0:
  pcalau12i $a0, %pc_hi20(ifunc0)
  addi.d $a0, $a0, %pc_lo12(ifunc0)
.L1:
  pcalau12i $a1, %got_pc_hi20(ifunc1)
  ld.d $a1, $a1, %got_pc_lo12(ifunc1)

.globl func
func:
  ret

## Resolvers are after relaxed code, so their addresses shift due to relaxation.
## The IRELATIVE addends must be adjusted accordingly.
.globl ifunc0, ifunc1
.type ifunc0, @gnu_indirect_function
.type ifunc1, @gnu_indirect_function
ifunc0:
  ret
ifunc1:
  ret
