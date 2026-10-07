## Ensure that out-of-range conditional tail calls can still reach the patched
## original entry in compact mode. Check both CONDBR19 and TSTBR14 relocations
## in both directions (+/- displacements).

# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-unknown %s -o %t.o
# RUN: ld.lld --emit-relocs --section-start=.text=0x300000 %t.o -o %t.exe
# RUN: llvm-bolt --compact-code-model --skip-funcs=_start --align-text=0x1000 \
# RUN:   --custom-allocation-vma=0x400000 %t.exe -o %t.bolt
# RUN: llvm-objdump -d %t.exe 2>&1 | FileCheck %s --check-prefix=NONBOLTED_FORWARD
# RUN: llvm-objdump -d %t.bolt 2>&1 | FileCheck %s --check-prefix=BOLTED_FORWARD

# NONBOLTED_FORWARD: [[#%x,BAR:]] <bar>:
# NONBOLTED_FORWARD: {{.*}} <_start>:
# NONBOLTED_FORWARD: {{.*}} b.ge 0x[[#BAR]] <bar>
# NONBOLTED_FORWARD: {{.*}} tbz x10, #0x20, 0x[[#BAR]] <bar>

## The relocated function is over 1MB ahead of the branch sites, so none of
## the conditional branches can reach it directly. The original entry must
## contain a patch that redirects execution to the relocated function.
# BOLTED_FORWARD: [[#%x,BAROLD:]] <bar.org.0>:
# BOLTED_FORWARD-NEXT: {{.*}} adrp x16, 0x[[#%x,RELOC:]] <bar>
# BOLTED_FORWARD-NEXT: {{.*}} add x16, x16, #0x0
# BOLTED_FORWARD-NEXT: {{.*}} br x16
# BOLTED_FORWARD: {{.*}} <_start>:
# BOLTED_FORWARD-NEXT: {{.*}} b.ge 0x[[#BAROLD]] <bar.org.0>
# BOLTED_FORWARD-NEXT: {{.*}} tbz x10, #0x20, 0x[[#BAROLD]] <bar.org.0>
# BOLTED_FORWARD: [[#RELOC]] <bar>:

# RUN: ld.lld --emit-relocs --section-start=.text=0x502000 %t.o -o %t.exe
# RUN: llvm-bolt --compact-code-model --skip-funcs=_start --align-text=0x1000 \
# RUN:   --custom-allocation-vma=0x400000 %t.exe -o %t.bolt
# RUN: llvm-objdump -d %t.exe 2>&1 | FileCheck %s --check-prefix=NONBOLTED_BACKWARD
# RUN: llvm-objdump -d %t.bolt 2>&1 | FileCheck %s --check-prefix=BOLTED_BACKWARD

# NONBOLTED_BACKWARD: [[#%x,BAR:]] <bar>:
# NONBOLTED_BACKWARD: {{.*}} <_start>:
# NONBOLTED_BACKWARD: {{.*}} b.ge 0x[[#BAR]] <bar>
# NONBOLTED_BACKWARD: {{.*}} tbz x10, #0x20, 0x[[#BAR]] <bar>

## The relocated function is over 1MB behind the branch sites. All conditional
## branches must still reach the original entry and its patch.
# BOLTED_BACKWARD: [[#%x,BAROLD:]] <bar.org.0>:
# BOLTED_BACKWARD-NEXT: {{.*}} adrp x16, 0x[[#%x,RELOC:]] <bar>
# BOLTED_BACKWARD-NEXT: {{.*}} add x16, x16, #0x0
# BOLTED_BACKWARD-NEXT: {{.*}} br x16
# BOLTED_BACKWARD: {{.*}} <_start>:
# BOLTED_BACKWARD-NEXT: {{.*}} b.ge 0x[[#BAROLD]] <bar.org.0>
# BOLTED_BACKWARD-NEXT: {{.*}} tbz x10, #0x20, 0x[[#BAROLD]] <bar.org.0>
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
  b.ge bar
  tbz x10, #32, bar
  ret
  .size _start, .-_start
