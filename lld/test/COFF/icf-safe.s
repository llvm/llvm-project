# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-win32 %s -o %t1.obj
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-win32 %S/Inputs/icf-safe.s -o %t2.obj
# RUN: lld-link /dll /noentry /out:%t.dll /verbose /opt:noref,icf %t1.obj %t2.obj 2>&1 | FileCheck %s
# RUN: lld-link /dll /noentry /out:%t.dll /verbose /opt:noref,icf /export:g3 /export:g4 %t1.obj %t2.obj 2>&1 | FileCheck --check-prefix=EXPORT %s
# RUN: lld-link /dll /noentry /out:%t.dll /verbose /opt:noref,safeicf %t1.obj %t2.obj 2>&1 | FileCheck %s --check-prefix=SAFEICF

# CHECK-NOT: Selected
# CHECK: Selected g3
# CHECK-NEXT:   Removed g4
# CHECK: Selected f1
# CHECK-NEXT:   Removed f2
# CHECK-NEXT:   Removed f3
# CHECK-NEXT:   Removed f4
# CHECK-NOT: Removed
# CHECK-NOT: Selected

# EXPORT-NOT: Selected g3
# EXPORT-NOT: Selected g4

# SAFEICF-NOT: Selected
# SAFEICF: Selected g3
# SAFEICF-NEXT:   Removed g4
# SAFEICF: Selected f3
# SAFEICF-NEXT:   Removed f4
# SAFEICF-NOT: Removed
# SAFEICF-NOT: Selected

.section .rdata,"dr",one_only,g1
.globl g1
g1:
.byte 1

.section .rdata,"dr",one_only,g2
.globl g2
g2:
.byte 1

.section .rdata,"dr",one_only,g3
.globl g3
g3:
.byte 2

.section .rdata,"dr",one_only,g4
.globl g4
g4:
.byte 2

.section .text,"xr",one_only,f1
.globl f1
f1:
 nop

.section .text,"xr",one_only,f2
.globl f2
f2:
 nop

.section .text,"xr",one_only,f3
.globl f3
f3:
 nop

.section .text,"xr",one_only,f4
.globl f4
f4:
 nop

.addrsig
.addrsig_sym g1
.addrsig_sym g2
.addrsig_sym f1
.addrsig_sym f2
