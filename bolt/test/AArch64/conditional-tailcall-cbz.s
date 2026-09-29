## Check support for conditional tail calls, ensure that conditional branches with
## CBZ/CBNZ are correctly relocated. Assume that everything is in range with 
## --no-huge-pages. 

# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-unknown %s -o %t.o
# RUN: ld.lld --emit-relocs %t.o -o %t.exe
# RUN: llvm-bolt --no-huge-pages --skip-funcs=_start %t.exe -o %t.bolt
# RUN: llvm-objdump -d %t.exe 2>&1 | FileCheck %s --check-prefix=NONBOLTED
# RUN: llvm-objdump -d %t.bolt 2>&1 | FileCheck %s --check-prefix=BOLTED

# NONBOLTED: [[#%x,BAR:]] <bar>:
# NONBOLTED: {{.*}} <_start>: 
# NONBOLTED: {{.*}} cbz w0, 0x[[#BAR]] <bar> 
# NONBOLTED: {{.*}} cbz x21, 0x[[#BAR]] <bar> 
# NONBOLTED: {{.*}} cbnz x10, 0x[[#BAR]] <bar> 
# NONBOLTED: {{.*}} cbnz wzr, 0x[[#BAR]] <bar> 

# BOLTED: [[#%x,BARORG:]] <bar.org.0>: 
# BOLTED: [[#BARORG]]: {{.*}} adrp x16, 0x[[#%x,BARNEW:]] <bar> 
# BOLTED: {{.*}} <_start>: 
# BOLTED: {{.*}} cbz w0, 0x[[#BARNEW]] <bar>
# BOLTED: {{.*}} cbz x21, 0x[[#BARNEW]] <bar> 
# BOLTED: {{.*}} cbnz x10, 0x[[#BARNEW]] <bar> 
# BOLTED: {{.*}} cbnz wzr, 0x[[#BARNEW]] <bar> 
# BOLTED: [[#BARNEW]] <bar>: 

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
  cbz w0, bar 
  cbz x21, bar

  cbnz x10, bar
  cbnz wzr, bar 
  ret
  .size _start, .-_start
  