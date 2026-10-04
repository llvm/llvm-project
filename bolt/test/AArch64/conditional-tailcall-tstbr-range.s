## Ensure that conditional calls using the TSTBR14 relocation type are only relocated 
## if the target is range. Do not relocate the instruction if the target is out of range. 
## Ensure that this applies in both directions (+/- displacements). 

# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-unknown %s -o %t.o
# RUN: ld.lld --emit-relocs --section-start=.text=0x3f8ff0 %t.o -o %t.exe
# RUN: llvm-bolt --skip-funcs=_start --align-text=0x1000 %t.exe -o %t.bolt
# RUN: llvm-objdump -d %t.exe 2>&1 | FileCheck %s --check-prefix=NONBOLTED_FORWARD
# RUN: llvm-objdump -d %t.bolt 2>&1 | FileCheck %s --check-prefix=BOLTED_FORWARD

# NONBOLTED_FORWARD: [[#%x,BAR:]] <bar>: 
# NONBOLTED_FORWARD: {{.*}} <_start>: 
# NONBOLTED_FORWARD: {{.*}} tbz x10, #0x20, 0x[[#BAR]] <bar> 
# NONBOLTED_FORWARD: {{.*}} tbz x10, #0x20, 0x[[#BAR]] <bar> 

# BOLTED_FORWARD: [[#%x,BAROLD:]] <bar.org.0>: 
# BOLTED_FORWARD: {{.*}} adrp x16, 0x[[#%x,RELOC:]] <bar>
# BOLTED_FORWARD: {{.*}} <_start>: 
## The first branch is out of range of the relocated function and hence points to the 
## original function, whereas the second is within range of the relocated function 
## and hence points to the newly relocated function. 
# BOLTED_FORWARD: {{.*}} tbz x10, #0x20, 0x[[#BAROLD]] <bar.org.0>
# BOLTED_FORWARD: {{.*}} tbz x10, #0x20, 0x[[#RELOC]] <bar> 
# BOLTED_FORWARD: [[#RELOC]] <bar>: 

# RUN: ld.lld --emit-relocs --section-start=.text=0x408ff0 %t.o -o %t.exe
# RUN: llvm-bolt --skip-funcs=_start --align-text=0x1000 \
# RUN:   --custom-allocation-vma=0x400000 %t.exe -o %t.bolt
# RUN: llvm-objdump -d %t.exe 2>&1 | FileCheck %s --check-prefix=NONBOLTED_BACKWARD
# RUN: llvm-objdump -d %t.bolt 2>&1 | FileCheck %s --check-prefix=BOLTED_BACKWARD

# NONBOLTED_BACKWARD: [[#%x,BAR:]] <bar>: 
# NONBOLTED_BACKWARD: {{.*}} <_start>: 
# NONBOLTED_BACKWARD: {{.*}} tbz x10, #0x20, 0x[[#BAR]] <bar> 
# NONBOLTED_BACKWARD: {{.*}} tbz x10, #0x20, 0x[[#BAR]] <bar> 

# BOLTED_BACKWARD: [[#%x,BAROLD:]] <bar.org.0>: 
# BOLTED_BACKWARD: {{.*}} adrp x16, 0x[[#%x,RELOC:]] <bar>
# BOLTED_BACKWARD: {{.*}} <_start>: 
## The first branch is in range of the relocated function and hence is relocated to the 
## relocated function, whereas the second is not and so points to the original function. 
# BOLTED_BACKWARD: {{.*}} tbz x10, #0x20, 0x[[#RELOC]] <bar>
# BOLTED_BACKWARD: {{.*}} tbz x10, #0x20, 0x[[#BAROLD]] <bar.org.0> 
# BOLTED_BACKWARD: [[#RELOC]] <bar>: 

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
  tbz x10, #32, bar
  tbz x10, #32, bar
  ret
  .size _start, .-_start
