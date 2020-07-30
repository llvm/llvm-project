# REQUIRES: ppc
# RUN: llvm-mc -filetype=obj -triple=powerpc %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readobj -r %t | FileCheck --check-prefix=RELOC %s
# RUN: llvm-readelf -S -s %t | FileCheck --check-prefixes=SEC,SYM %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

# RELOC:      .rela.dyn {
# RELOC-NEXT:   0x10020118 R_PPC_IRELATIVE - 0x100100E0
# RELOC-NEXT: }

# SEC: .rela.dyn RELA 100000d4 0000d4 00000c
# SYM: 100100e0 0 FUNC GLOBAL DEFAULT {{.*}} func

# CHECK-LABEL: <func_resolver>:
# CHECK-NEXT:    100100e0:
# CHECK-LABEL: <_start>:
# CHECK-NEXT:    bl 0x100100f8
## .rela.plt = 0x100000d4 = 65536*4096+212
## end(.rela.plt) = 0x100000d4+0xc = 65536*4096+224
# CHECK-NEXT:    lis 9, 4096
# CHECK-NEXT:    lis 8, 4096
# CHECK-NEXT:    addi 9, 9, 212
# CHECK-NEXT:    addi 8, 8, 224

.globl func
.type func, @gnu_indirect_function
func:
.globl func_resolver
.type func_resolver, @function
func_resolver:
  blr

.globl _start
_start:
  bl func

  lis 9, __rela_iplt_start@ha
  lis 8, __rela_iplt_end@ha
  la 9, __rela_iplt_start@l(9)
  la 8, __rela_iplt_end@l(8)
