## Check support for conditional tail calls, ensure that conditional branches with
## B.cond are correctly relocated. Assume that everything is in range with 
## --no-huge-pages. 

# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown %s -o %t.o
# RUN: ld.lld --emit-relocs %t.o -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt --skip-funcs=_start --no-huge-pages 
# RUN: llvm-objdump -d %t.exe 2>&1 | FileCheck %s --check-prefix=NONBOLTED
# RUN: llvm-objdump -d %t.bolt 2>&1 | FileCheck %s --check-prefix=BOLTED

# NONBOLTED: [[#%x,FOONOBOLT:]] <foo>: 
# NONBOLTED: {{.*}} <_start>:
# NONBOLTED: {{.*}} b.ge 0x[[#FOONOBOLT]] <foo> 
# NONBOLTED: {{.*}} b.pl 0x[[#FOONOBOLT]] <foo>

# BOLTED: {{.*}} <foo.org.0>: 
# BOLTED: {{.*}} adrp x16, 0x[[#%x,FOO:]] <foo> 
# BOLTED: {{.*}} <_start>: 
# BOLTED: {{.*}} b.ge 0x[[#FOO]] <foo> 
# BOLTED: {{.*}} b.pl 0x[[#FOO]] <foo> 
# BOLTED: [[#FOO]] <foo>:  

    .type foo,@function
    .globl foo
foo: 
  .rept 3
    nop
  .endr  
  ret
  .size foo, .-foo

    .type _start,@function
    .globl _start
_start: 
  b.ge foo
  b.pl foo
  ret
  .size _start, .-_start
