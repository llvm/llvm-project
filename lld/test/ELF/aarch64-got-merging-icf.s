## REQUIRES: aarch64
## Check that symbols that ICF assumes to be the same get a single GOT entry

# RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t
# RUN: llvm-mc -filetype=obj -crel -triple=aarch64 %s -o %tcrel
# RUN: ld.lld %t -o %t2 --icf=all
# RUN: ld.lld %tcrel -o %tcrel2 --icf=all

# RUN: llvm-objdump --section-headers %t2 | FileCheck %s --check-prefix=EXE
# RUN: llvm-objdump --section-headers %tcrel2 | FileCheck %s --check-prefix=EXE

# RUN: ld.lld -shared %t -o %t3 --icf=all
# RUN: ld.lld -shared %tcrel -o %tcrel3 --icf=all

# RUN: llvm-objdump --section-headers %t3 | FileCheck %s --check-prefix=DSO
# RUN: llvm-objdump --section-headers %tcrel3 | FileCheck %s --check-prefix=DSO

## All global g* symbols should merge into a single GOT entry while non-global
## gets its own GOT entry.
# EXE: {{.*}}.got 00000010{{.*}}

## When symbols are preemptible in DSO mode, GOT entries wouldn't be merged
# DSO: {{.*}}.got 00000028{{.*}}

.addrsig

callee:
ret

.macro f, index, isglobal

# (Kept unique) first instruction of the GOT code sequence
.section .text.f1_\index,"ax",@progbits
f1_\index:
adrp x0, :got:g\index
mov x1, #\index
b f2_\index

# Folded, second instruction of the GOT code sequence
.section .text.f2_\index,"ax",@progbits
f2_\index:
ldr x0, [x0, :got_lo12:g\index] 
b callee

# Folded
.ifnb \isglobal
.globl g\index
.endif
.section .rodata.g\index,"a",@progbits
g_\index:
.long 111
.long 122

g\index:
.byte 123

.section .text._start,"ax",@progbits
bl f1_\index

.endm

## Another set of sections merging: g1 <- g2. Linker should be able to
## resolve both g1 and g2 to g0 based on ICF on previous sections.

.section .text.t1_0,"ax",@progbits
t1_0:
adrp x2, :got:g1
mov x3, #1
b t2_0

.section .text.t2_0,"ax",@progbits
t2_0:
ldr x2, [x2, :got_lo12:g1]
b callee

.section .text.t1_1,"ax",@progbits
t1_1:
adrp x2, :got:g2
mov x3, #2
b t2_1

.section .text.t2_1,"ax",@progbits
t2_1:
ldr x2, [x2, :got_lo12:g2]
b callee

.section .text._start,"ax",@progbits
.globl _start
_start:

f 0 1
f 1 1
f 2 1
f 3 1
f 4
