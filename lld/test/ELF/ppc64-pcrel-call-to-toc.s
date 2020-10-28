# REQUIRES: ppc
# RUN: echo 'SECTIONS { \
# RUN:   .text_func   0x10010000 : { *(.text_func) } \
# RUN:   .text_callee 0x10020000 : { *(.text_callee) } \
# RUN:   .text_caller 0x10030000 : { *(.text_caller) } \
# RUN:   }' > %t.script

# RUN: llvm-mc -filetype=obj -triple=powerpc64le %s -o %t.o
# RUN: ld.lld -T %t.script %t.o -o %t
# RUN: llvm-readelf -s %t | FileCheck %s --check-prefix=SYMBOL
# RUN: llvm-objdump -d --no-show-raw-insn --mcpu=pwr10 %t | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=powerpc64 %s -o %t.o
# RUN: ld.lld -T %t.script %t.o -o %t
# RUN: llvm-readelf -s %t | FileCheck %s --check-prefix=SYMBOL
# RUN: llvm-objdump -d --no-show-raw-insn --mcpu=pwr10 %t | FileCheck %s

## When a function without TOC accesses a function using TOC, an r12 setup stub
## is inserted

# SYMBOL:      1: 0000000010020000 0 NOTYPE LOCAL DEFAULT [<other: 0x60>] 2 callee
# SYMBOL-NEXT: 2: 0000000010030000 0 NOTYPE LOCAL DEFAULT [<other: 0x20>] 3 caller
# SYMBOL-NEXT: 3: 0000000010010000 0 NOTYPE LOCAL DEFAULT 1 func
# SYMBOL:      6: 000000001003000c 16 FUNC LOCAL DEFAULT 3 __gep_setup_callee

# CHECK-LABEL: <func>:
# CHECK-NEXT:  blr

# CHECK-LABEL: <callee>:
# CHECK:       bl 0x10010000
# CHECK-NEXT:  addis 4, 2, -1
# CHECK-NEXT:  lwz 4, 32744(4)
# CHECK-NEXT:  blr

# CHECK-LABEL: <caller>:
# CHECK-NEXT:  bl 0x1003000c
# CHECK-NEXT:  blr

# CHECK-LABEL: <__gep_setup_callee>:
# CHECK-NEXT:  paddi 12, 0, -65548, 1
# CHECK-NEXT:  mtctr 12
# CHECK-NEXT:  bctr

.section .text_func, "ax", %progbits
func:
  blr

.section .text_callee, "ax", %progbits
callee:
.Lfunc_gep1:
  addis 2, 12, .TOC.-.Lfunc_gep1@ha
  addi 2, 2, .TOC.-.Lfunc_gep1@l
.Lfunc_lep1:
  .localentry callee, .Lfunc_lep1-.Lfunc_gep1
  bl func
  addis 4, 2, global@toc@ha
  lwz 4, global@toc@l(4)
  blr

.section .text_caller, "ax", %progbits
caller:
  .localentry caller, 1
  bl callee@notoc
  blr
global:
  .long	0
  .size	global, 4
