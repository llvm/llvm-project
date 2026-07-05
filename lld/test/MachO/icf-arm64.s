# REQUIRES: aarch64
# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin19.0.0 %t/main.s -o %t/main.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin19.0.0 %t/f2.s -o %t/f2.o
# RUN: %lld -arch arm64 -lSystem --icf=all -o %t/main %t/main.o %t/f2.o
# RUN: llvm-objdump -d --syms --print-imm-hex %t/main | FileCheck %s

# CHECK-LABEL: SYMBOL TABLE:
# CHECK: [[#%x,F1_REF:]]                 g     F __TEXT,__text _f1
# CHECK: [[#%x,F1_REF:]]                 g     F __TEXT,__text _f2

# CHECK-LABEL: Disassembly of section __TEXT,__text:
# CHECK:        <_main>:
# CHECK: bl 0x[[#%x,F1_REF:]]
# CHECK: bl 0x[[#%x,F1_REF:]]

#--- main.s

.subsections_via_symbols

.literal16
.p2align 3
L_align16:
.quad 0xffffffffffffffff
.short 0xaaaa
.short 0xaaaa
.space 4, 0xaa

.literal8
.p2align 3
L_align8:
.quad 0xeeeeeeeeeeeeeeee

.literal4
.p2align 2
L_align4:
.short 0xbbbb
.short 0xbbbb


.text
.p2align 2

.globl _main, _f1, _f2

## Test that loading from __literalN sections at non-literal boundaries
## doesn't confuse ICF. This function should be folded with the identical
## _f2 in f2 (which uses literals of the same value in a different isec).
_f1:
  adrp x9, L_align16@PAGE + 4
  add x9, x9, L_align16@PAGEOFF + 4
  ldr x10, [x9]

  adrp x9, L_align8@PAGE + 4
  add x9, x9, L_align8@PAGEOFF + 4
  ldr w11, [x9]

  adrp x9, L_align4@PAGE + 2
  add x9, x9, L_align4@PAGEOFF + 2
  ldrh w12, [x9]

  ret

_main:
  bl _f1
  bl _f2

#--- f2.s

.subsections_via_symbols

.literal16
.p2align 3
L_align16:
.quad 0xffffffffffffffff
.short 0xaaaa
.short 0xaaaa
.space 4, 170

.literal8
.p2align 3
L_align8:
.quad 0xeeeeeeeeeeeeeeee

.literal4
.p2align 2
L_align4:
.short 0xbbbb
.short 0xbbbb

.text
.p2align 2

.globl _f2
_f2:
  adrp x9, L_align16@PAGE + 4
  add x9, x9, L_align16@PAGEOFF + 4
  ldr x10, [x9]

  adrp x9, L_align8@PAGE + 4
  add x9, x9, L_align8@PAGEOFF + 4
  ldr w11, [x9]

  adrp x9, L_align4@PAGE + 2
  add x9, x9, L_align4@PAGEOFF + 2
  ldrh w12, [x9]

  ret
