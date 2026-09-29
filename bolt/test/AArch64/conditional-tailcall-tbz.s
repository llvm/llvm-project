## Check support for conditional tail calls, ensure that conditional branches with
## TBZ/TBNZ are correctly relocated. Assume that everything is in range by 
## aligning the text sections of the binaries. 

# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-unknown %s -o %t.o
# RUN: ld.lld --emit-relocs --section-start=.text=0x3ff000 %t.o -o %t.exe
# RUN: llvm-bolt --skip-funcs=_start --align-text=0x1000 %t.exe -o %t.bolt
# RUN: llvm-objdump -d %t.exe 2>&1 | FileCheck %s --check-prefix=NONBOLTED
# RUN: llvm-objdump -d %t.bolt 2>&1 | FileCheck %s --check-prefix=BOLTED

# NONBOLTED: {{0*}}3ff000 <bar>: 
# NONBOLTED: {{.*}} <_start>: 
# NONBOLTED: {{.*}} tbz w0, #0x0, 0x3ff000 <bar> 
# NONBOLTED: {{.*}} tbz x10, #0x20, 0x3ff000 <bar>
# NONBOLTED: {{.*}} tbnz w21, #0x1f, 0x3ff000 <bar>
# NONBOLTED: {{.*}} tbnz xzr, #0x3f, 0x3ff000 <bar>

# BOLTED: {{0*}}3ff000 <bar.org.0>:
# BOLTED: {{0*}}3ff000: {{.*}} adrp x16, 0x[[#%x,BAR:]] <bar>
# BOLTED: {{.*}} <_start>:
# BOLTED: {{.*}} tbz w0, #0x0, 0x[[#BAR]] <bar> 
# BOLTED: {{.*}} tbz x10, #0x20, 0x[[#BAR]] <bar>
# BOLTED: {{.*}} tbnz w21, #0x1f, 0x[[#BAR]] <bar>
# BOLTED: {{.*}} tbnz xzr, #0x3f, 0x[[#BAR]] <bar>
# BOLTED: [[#BAR]] <bar>:

    .type bar,@function
    .globl bar
bar: 
  .rept 3
    nop
  .endr  
  ret
  .size bar, .-bar

    .type _start,@function
    .globl _start
_start: 
  tbz w0, #0, bar
  tbz x10, #32, bar

  tbnz w21, #31, bar
  tbnz xzr, #63, bar
  ret
  .size _start, .-_start
  