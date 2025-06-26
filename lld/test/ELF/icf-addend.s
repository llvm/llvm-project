# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t

#--- trivial-relocation.s
# Tests following for trivial relocations:
# 1. Merging two equivalent sections is allowed but we must not merge their symbols if addends are different.
# 2. Local symbols should not be merged together even though their sections can be merged together.

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux trivial-relocation.s -o trivial.o
# RUN: ld.lld trivial.o -o /dev/null --icf=all --print-icf-sections | FileCheck %s --check-prefix=TRIVIAL

# TRIVIAL: selected section {{.*}}:(.rodata.sec1)
# TRIVIAL-NEXT:   removing identical section {{.*}}:(.rodata.sec2)
# TRIVIAL-NEXT: selected section {{.*}}:(.text.f1)
# TRIVIAL-NEXT:   removing identical section {{.*}}:(.text.f2)
# TRIVIAL-NEXT:   removing identical section {{.*}}:(.text.f1_local)
# TRIVIAL-NEXT:   removing identical section {{.*}}:(.text.f2_local)

.addrsig

.globl x_glob, y_glob

.section .rodata.sec1,"a",@progbits
x_glob:
.long 11
y_glob:
.long 12

.section .rodata.sec2,"a",@progbits
x:
.long 11
y:
.long 12

.section .text.f1,"ax",@progbits
f1:
movq x_glob+4(%rip), %rax

.section .text.f2,"ax",@progbits
f2:
movq y_glob(%rip), %rax

.section .text.f1_local,"ax",@progbits
f1_local:
movq x+4(%rip), %rax

.section .text.f2_local,"ax",@progbits
f2_local:
movq y(%rip), %rax

.section .text._start,"ax",@progbits
.globl _start
_start:
call f1
call f2

#--- non-trivial-relocation.s
# Tests following for non-trivial relocations:
# 1. We must not merge sections if addends are different.
# 2. We must not merge sections pointing to local and global symbols.

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux non-trivial-relocation.s -o non-trivial.o
# RUN: ld.lld non-trivial.o -o /dev/null --icf=all --print-icf-sections | FileCheck %s --check-prefix=NONTRIVIAL

# NONTRIVIAL: selected section {{.*}}:(.rodata.sec1)
# NONTRIVIAL-NEXT:   removing identical section {{.*}}:(.rodata.sec2)
# NONTRIVIAL-NEXT: selected section {{.*}}:(.text.f1_local)
# NONTRIVIAL-NEXT:   removing identical section {{.*}}:(.text.f2_local)

.addrsig

.globl x_glob, y_glob

.section .rodata.sec1,"a",@progbits
x_glob:
.long 11
y_glob:
.long 12

.section .rodata.sec2,"a",@progbits
x:
.long 11
y:
.long 12

.section .text.f1,"ax",@progbits
f1:
movq x_glob+4@GOTPCREL(%rip), %rax

.section .text.f2,"ax",@progbits
f2:
movq y_glob@GOTPCREL(%rip), %rax

.section .text.f1_local,"ax",@progbits
f1_local:
movq x+4(%rip), %rax

.section .text.f2_local,"ax",@progbits
f2_local:
movq y(%rip), %rax

.section .text._start,"ax",@progbits
.globl _start
_start:
call f1
call f2
