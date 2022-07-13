# REQUIRES: ppc
# RUN: echo 'SECTIONS { \
# RUN:   .text_default_stother0 0x10010000: { *(.text_default_stother0) } \
# RUN:   .text_default_stother1 0x10020000: { *(.text_default_stother1) } \
# RUN:   .text_hidden_stother0 0x10030000: { *(.text_hidden_stother0) } \
# RUN:   .text_hidden_stother1 0x10040000: { *(.text_hidden_stother1) } \
# RUN:   }' > %t.script

# RUN: llvm-mc -filetype=obj -triple=powerpc64le -defsym HIDDEN=1 %s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=powerpc64le %p/Inputs/ppc64-callee-global-hidden.s -o %t2.o
# RUN: ld.lld -T %t.script -shared %t1.o %t2.o -o %t.so
# RUN: llvm-readelf -s %t.so | FileCheck %s --check-prefix=SYMBOL
# RUN: llvm-objdump -d --no-show-raw-insn %t.so | FileCheck --check-prefix=CHECK --check-prefix=CHECK-HIDDEN %s

# RUN: llvm-mc -filetype=obj -triple=powerpc64le -defsym GLOBAL=1 %s -o %t3.o
# RUN: ld.lld -T %t.script %t3.o -o %t
# RUN: llvm-readelf -s %t | FileCheck %s --check-prefix=SYMBOL-GLOBAL
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=powerpc64 -defsym HIDDEN=1 %s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=powerpc64 %p/Inputs/ppc64-callee-global-hidden.s -o %t2.o
# RUN: ld.lld -T %t.script -shared %t1.o %t2.o -o %t.so
# RUN: llvm-readelf -s %t.so | FileCheck %s --check-prefix=SYMBOL
# RUN: llvm-objdump -d --no-show-raw-insn %t.so | FileCheck --check-prefix=CHECK --check-prefix=CHECK-HIDDEN %s

# RUN: llvm-mc -filetype=obj -triple=powerpc64 -defsym GLOBAL=1 %s -o %t3.o
# RUN: ld.lld -T %t.script %t3.o -o %t
# RUN: llvm-readelf -s %t | FileCheck %s --check-prefix=SYMBOL-GLOBAL
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

# SYMBOL:      2: 0000000010010000 0 NOTYPE LOCAL DEFAULT 5 callee1_stother0_default
# SYMBOL-NEXT: 3: 0000000010010004 0 NOTYPE LOCAL DEFAULT [<other: 0x20>] 5 caller1
# SYMBOL-NEXT: 4: 0000000010020000 0 NOTYPE LOCAL DEFAULT 6 func_local
# SYMBOL-NEXT: 5: 0000000010020004 0 NOTYPE LOCAL DEFAULT [<other: 0x20>] 6 callee2_stother1_default
# SYMBOL-NEXT: 6: 000000001002000c 0 NOTYPE LOCAL DEFAULT [<other: 0x20>] 6 caller2
# SYMBOL-NEXT: 7: 0000000010030000 0 NOTYPE LOCAL DEFAULT [<other: 0x20>] 7 caller3
# SYMBOL-NEXT: 8: 0000000010040000 0 NOTYPE LOCAL DEFAULT [<other: 0x20>] 8 caller4
# SYMBOL-NEXT: 9: 0000000010040008 0 NOTYPE LOCAL DEFAULT 9 func_extern
# SYMBOL-NEXT: 10: 000000001004000c 0 NOTYPE LOCAL HIDDEN 9 callee3_stother0_hidden
# SYMBOL-NEXT: 11: 0000000010040010 0 NOTYPE LOCAL HIDDEN [<other: 0x22>] 9 callee4_stother1_hidden

# SYMBOL-GLOBAL:      2: 0000000010010004 0 NOTYPE LOCAL DEFAULT [<other: 0x20>] 1 caller1
# SYMBOL-GLOBAL-NEXT: 3: 0000000010020000 0 NOTYPE LOCAL DEFAULT 2 func_local
# SYMBOL-GLOBAL-NEXT: 4: 000000001002000c 0 NOTYPE LOCAL DEFAULT [<other: 0x20>] 2 caller2
# SYMBOL-GLOBAL-NEXT: 5: 0000000010010000 0 NOTYPE GLOBAL DEFAULT 1 callee1_stother0_default
# SYMBOL-GLOBAL-NEXT: 6: 0000000010020004 0 NOTYPE GLOBAL DEFAULT [<other: 0x20>] 2 callee2_stother1_default

# CHECK-LABEL: <callee1_stother0_default>:
# CHECK-NEXT:  10010000: blr

# CHECK-LABEL: <caller1>:
# CHECK:       10010004: bl 0x10010000
# CHECK-NEXT:  10010008: b 0x10010000
.section .text_default_stother0, "ax", %progbits
.ifdef GLOBAL
.globl callee1_stother0_default
.endif
callee1_stother0_default:
  blr
caller1:
  .localentry caller1, 1
  ## nop is not needed after bl for R_PPC64_REL24_NOTOC
  bl callee1_stother0_default@notoc
  b callee1_stother0_default@notoc

# CHECK-LABEL: <func_local>:
# CHECK-NEXT:  10020000: blr

# CHECK-LABEL: <callee2_stother1_default>:
# CHECK-NEXT:  10020004: bl 0x10020000
# CHECK-NEXT:  10020008: blr

# CHECK-LABEL: <caller2>:
# CHECK:       1002000c: bl 0x10020004
# CHECK-NEXT:  10020010: b 0x10020004
.section .text_default_stother1, "ax", %progbits
func_local:
  blr
.ifdef GLOBAL
.globl callee2_stother1_default
.endif
callee2_stother1_default:
  .localentry callee2_stother1_default, 1
  ## nop is not needed after bl for R_PPC64_REL24_NOTOC
  bl func_local@notoc
  blr
caller2:
  .localentry caller2, 1
  ## nop is not needed after bl for R_PPC64_REL24_NOTOC
  bl callee2_stother1_default@notoc
  b callee2_stother1_default@notoc

# CHECK-HIDDEN-LABEL: <caller3>:
# CHECK-HIDDEN-NEXT:  10030000: bl 0x1004000c
# CHECK-HIDDEN-NEXT:  10030004: b 0x1004000c

# CHECK-HIDDEN-LABEL: <caller4>:
# CHECK-HIDDEN-NEXT:  10040000: bl 0x10040010
# CHECK-HIDDEN-NEXT:  10040004: b 0x10040010

# CHECK-HIDDEN-LABEL: <func_extern>:
# CHECK-HIDDEN-NEXT:  10040008: blr

# CHECK-HIDDEN-LABEL: <callee3_stother0_hidden>:
# CHECK-HIDDEN-NEXT:  1004000c: blr

# CHECK-HIDDEN-LABEL: <callee4_stother1_hidden>:
# CHECK-HIDDEN-NEXT:  10040010: bl 0x10040008
# CHECK-HIDDEN-NEXT:  10040014: blr
.ifdef HIDDEN
.section .text_hidden_stother0, "ax", %progbits
caller3:
  .localentry caller3, 1
  ## nop is not needed after bl for R_PPC64_REL24_NOTOC
  bl callee3_stother0_hidden@notoc
  b callee3_stother0_hidden@notoc

.section .text_hidden_stother1, "ax", %progbits
caller4:
  .localentry caller4, 1
  ## nop is not needed after bl for R_PPC64_REL24_NOTOC
  bl callee4_stother1_hidden@notoc
  b callee4_stother1_hidden@notoc
.endif
