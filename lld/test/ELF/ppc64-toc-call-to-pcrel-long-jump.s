# REQUIRES: ppc
# RUN: split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=powerpc64le %t/asm -o %t.o
# RUN: ld.lld -T %t/lts %t.o -o %t_le
# RUN: llvm-objdump --no-show-raw-insn -d %t_le | FileCheck %s
# RUN: llvm-readelf -s %t_le | FileCheck %s --check-prefix=SYM

# RUN: llvm-mc -filetype=obj -triple=powerpc64 %t/asm -o %t.o
# RUN: ld.lld -T %t/lts %t.o -o %t_be
# RUN: llvm-objdump --no-show-raw-insn -d %t_be | FileCheck %s
# RUN: llvm-readelf -s %t_be | FileCheck %s --check-prefix=SYM

# RUN: llvm-mc -filetype=obj -triple=powerpc64le %t/asm -o %t.o
# RUN: ld.lld -T %t/lts %t.o -o %t_le --no-power10-stubs
# RUN: llvm-objdump --no-show-raw-insn -d %t_le | FileCheck %s --check-prefix=NoP10
# RUN: llvm-readelf -s %t_le | FileCheck %s --check-prefix=SYM

# SYM:      Symbol table '.symtab' contains 9 entries:
# SYM:      1: 0000000010010000     0 NOTYPE  LOCAL  DEFAULT [<other: 0x20>]   1 callee
# SYM-NEXT: 2: 0000000010020008     0 NOTYPE  LOCAL  DEFAULT                  2 caller_close
# SYM-NEXT: 3: 0000000020020008     0 NOTYPE  LOCAL  DEFAULT [<other: 0x60>]   3 caller
# SYM-NEXT: 4: 0000000520020008     0 NOTYPE  LOCAL  DEFAULT                  4 caller_far
# SYM-NEXT: 5: 0000000520028040     0 NOTYPE  LOCAL  HIDDEN                   6 .TOC.
# SYM-NEXT: 6: 0000000010020020     8 FUNC    LOCAL  DEFAULT                  2 __toc_save_callee
# SYM-NEXT: 7: 0000000020020020    32 FUNC    LOCAL  DEFAULT                  3 __toc_save_callee
# SYM-NEXT: 8: 0000000520020020    32 FUNC    LOCAL  DEFAULT                  4 __toc_save_callee

#--- lts
PHDRS {
  callee PT_LOAD FLAGS(0x1 | 0x4);
  close PT_LOAD FLAGS(0x1 | 0x4);
  caller PT_LOAD FLAGS(0x1 | 0x4);
  far PT_LOAD FLAGS(0x1 | 0x4);
}
SECTIONS {
  .text_callee 0x10010000 : { *(.text_callee) } :callee
  .text_caller_close 0x10020000 : { *(.text_caller_close) } :close
  .text_caller 0x20020000 : { *(.text_caller) } :caller
  .text_caller_far 0x520020000 : { *(.text_caller_far) } :far
}

#--- asm
# CHECK-LABEL: <callee>:
# CHECK:         blr
.section .text_callee, "ax", %progbits
callee:
  .localentry callee, 1
  blr

# CHECK-LABEL: <caller_close>:
# CHECK:         bl 0x10020020
# CHECK-NEXT:    ld 2, 24(1)
# CHECK-NEXT:    blr
# CHECK-LABEL: <__toc_save_callee>:
# CHECK:         std 2, 24(1)
# CHECK-NEXT:    b 0x10010000
.section .text_caller_close, "ax", %progbits
.Lfunc_toc1:
  .quad .TOC.-.Lfunc_gep1
caller_close:
.Lfunc_gep1:
  ld 2, .Lfunc_toc1-.Lfunc_gep1(12)
  add 2, 2, 12
.Lfunc_lep1:
  .localentry caller, .Lfunc_lep1-.Lfunc_gep1
  bl callee
  nop
  blr

# CHECK-LABEL: <caller>:
# CHECK:         bl 0x20020020
# CHECK-NEXT:    ld 2, 24(1)
# CHECK-NEXT:    blr
# CHECK-LABEL: <__toc_save_callee>:
# CHECK:         std 2, 24(1)
# CHECK-NEXT:    paddi 12, 0, -268501028, 1
# CHECK-NEXT:    mtctr 12
# CHECK-NEXT:    bctr

# NoP10-LABEL: <caller>:
# NoP10:         bl 0x20020020
# NoP10-NEXT:    ld 2, 24(1)
# NoP10-NEXT:    blr
# NoP10-LABEL: <__toc_save_callee>:
# NoP10-NEXT:         std 2, 24(1)
# NoP10-NEXT:    addis 12, 2, -4098
# NoP10-NEXT:    addi 12, 12, 32704
# NoP10-NEXT:    mtctr 12
# NoP10-NEXT:    bctr
.section .text_caller, "ax", %progbits
.Lfunc_toc2:
  .quad .TOC.-.Lfunc_gep2
caller:
.Lfunc_gep2:
  ld 2, .Lfunc_toc2-.Lfunc_gep2(12)
  add 2, 2, 12
.Lfunc_lep2:
  .localentry caller, .Lfunc_lep2-.Lfunc_gep2
  bl callee
  nop
  blr

# CHECK-LABEL: <caller_far>:
# CHECK:         ld 2, -8(12)
# CHECK-NEXT:    add 2, 2, 12
# CHECK-NEXT:    bl 0x520020020
# CHECK-NEXT:    ld 2, 24(1)
# CHECK-NEXT:    blr
# CHECK-LABEL: <__toc_save_callee>:
# CHECK:         std 2, 24(1)
# CHECK-NEXT:    addis 12, 2, 0
# CHECK-NEXT:    ld 12, -32760(12)
# CHECK-NEXT:    mtctr 12
# CHECK-NEXT:    bctr
.section .text_caller_far, "ax", %progbits
.Lfunc_toc3:
  .quad .TOC.-.Lfunc_gep3
caller_far:
.Lfunc_gep3:
  ld 2, .Lfunc_toc3-.Lfunc_gep3(12)
  add 2, 2, 12
.Lfunc_lep3:
  .localentry caller, .Lfunc_lep3-.Lfunc_gep3
  bl callee
  nop
  blr
