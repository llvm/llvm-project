# REQUIRES: ppc
# RUN: echo 'SECTIONS { \
# RUN:   .text_start 0x10010300 : { *(.text_start) } \
# RUN:   }' > %t.script

# RUN: llvm-mc -filetype=obj -triple=powerpc64le %s --defsym T1=1 -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=powerpc64le %s --defsym T2=1 -o %t2.o
# RUN: llvm-mc -filetype=obj -triple=powerpc64le %s --defsym T3=1 -o %t3.o
# RUN: ld.lld --shared %t1.o -o %t1.so
# RUN: ld.lld -T %t.script %t1.so %t2.o -o %t2
# RUN: ld.lld -T %t.script %t1.so %t3.o -o %t3
# RUN: llvm-objdump -d --no-show-raw-insn %t2 | FileCheck %s --check-prefix=T2
# RUN: llvm-objdump -d --no-show-raw-insn %t3 | FileCheck %s --check-prefix=T3

# RUN: llvm-mc -filetype=obj -triple=powerpc64 %s --defsym T1=1 -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=powerpc64 %s --defsym T2=1 -o %t2.o
# RUN: llvm-mc -filetype=obj -triple=powerpc64 %s --defsym T3=1 -o %t3.o
# RUN: ld.lld --shared %t1.o -o %t1.so
# RUN: ld.lld -T %t.script %t1.so %t2.o -o %t2
# RUN: ld.lld -T %t.script %t1.so %t3.o -o %t3
# RUN: llvm-objdump -d --no-show-raw-insn %t2 | FileCheck %s --check-prefix=T2
# RUN: llvm-objdump -d --no-show-raw-insn %t3 | FileCheck %s --check-prefix=T3

.ifdef T1
.globl callee
callee:
  blr
.endif

# T2-LABEL: <p9codegen>:
# T2-NEXT:    10010300: addis 2, 12, 1
# T2-NEXT:    10010304: addi 2, 2, -32368
# T2-NEXT:    10010308: addis 4, 2, -1
# T2-NEXT:    1001030c: lwa 3, 32412(4)
# T2-NEXT:    10010310: bl 0x10010330
# T2-NEXT:    10010314: ld 2, 24(1)
# T2-NEXT:    10010318: blr

# T2-LABEL: <p10codegen>:
# T2-NEXT:    1001031c: plwa 3, 16(0), 1
# T2-NEXT:    10010324: bl 0x10010350
# T2-NEXT:    10010328: blr

# T2-LABEL: <__plt_callee>:
# T2-NEXT:    10010330: std 2, 24(1)
# T2-NEXT:    10010334: addis 12, 2, 0
# T2-NEXT:    10010338: ld 12, -32744(12)
# T2-NEXT:    1001033c: mtctr 12
# T2-NEXT:    10010340: bctr

# T2-LABEL: <__plt_pcrel_callee>:
# T2-NEXT:    10010350: pld 12, 344(0), 1
# T2-NEXT:    10010358: mtctr 12
# T2-NEXT:    1001035c: bctr
.ifdef T2
.section .text_start, "ax", %progbits
p9codegen:
.Lfunc_gep0:
  addis 2, 12, .TOC.-.Lfunc_gep0@ha
  addi 2, 2, .TOC.-.Lfunc_gep0@l
.Lfunc_lep0:
  .localentry	p9codegen, .Lfunc_lep0-.Lfunc_gep0
  addis 4, 2, Global@toc@ha
  lwa 3, Global@toc@l(4)
  bl callee
  nop
  blr
p10codegen:
  .localentry	main, 1
  plwa 3, Global@PCREL(0), 1
  bl callee@notoc
  blr
.globl Global
Global:
  .long	10
  .size	Global, 4
.endif

# T3-LABEL: <p10codegen>:
# T3-NEXT:    10010300: plwa 3, 44(0), 1
# T3-NEXT:    10010308: bl 0x10010330
# T3-NEXT:    1001030c: blr

# T3-LABEL: <p9codegen>:
# T3-NEXT:    10010310: addis 2, 12, 1
# T3-NEXT:    10010314: addi 2, 2, -32392
# T3-NEXT:    10010318: addis 4, 2, -1
# T3-NEXT:    1001031c: lwa 3, 32420(4)
# T3-NEXT:    10010320: bl 0x10010350
# T3-NEXT:    10010324: ld 2, 24(1)
# T3-NEXT:    10010328: blr

# T3-LABEL: <__plt_pcrel_callee>:
# T3-NEXT:    10010330: pld 12, 368(0), 1
# T3-NEXT:    10010338: mtctr 12
# T3-NEXT:    1001033c: bctr

# T3-LABEL: <__plt_callee>:
# T3-NEXT:    10010350: std 2, 24(1)
# T3-NEXT:    10010354: addis 12, 2, 0
# T3-NEXT:    10010358: ld 12, -32744(12)
# T3-NEXT:    1001035c: mtctr 12
# T3-NEXT:    10010360: bctr
.ifdef T3
.section .text_start, "ax", %progbits
p10codegen:
  .localentry   main, 1
  plwa 3, Global@PCREL(0), 1
  bl callee@notoc
  blr
p9codegen:
.Lfunc_gep0:
  addis 2, 12, .TOC.-.Lfunc_gep0@ha
  addi 2, 2, .TOC.-.Lfunc_gep0@l
.Lfunc_lep0:
  .localentry   p9codegen, .Lfunc_lep0-.Lfunc_gep0
  addis 4, 2, Global@toc@ha
  lwa 3, Global@toc@l(4)
  bl callee
  nop
  blr
.globl Global
Global:
  .long 10
  .size Global, 4
.endif
