# REQUIRES: x86
## Sections reference each other, so folding one pair can enable another.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 self.s -o self.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 cycle1.s -o cycle1.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 cycle2.s -o cycle2.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 cycle-diff.s -o cycle-diff.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 chain.s -o chain.o

## f1 and f2 reference a symbol they define themselves.
# RUN: ld.lld self.o -o self --icf=all --print-icf-sections | \
# RUN:   FileCheck %s --implicit-check-not=section
# CHECK:      selected section {{.*}}self.o:(.text.f1)
# CHECK-NEXT:   removing identical section {{.*}}self.o:(.text.f2)

## f1 and f2 call each other, so neither can be decided before the other.
# RUN: ld.lld cycle1.o cycle2.o -o cycle --icf=all --print-icf-sections | \
# RUN:   FileCheck %s --check-prefix=CYCLE --implicit-check-not=section
# CYCLE:      selected section {{.*}}cycle1.o:(.text.f1)
# CYCLE-NEXT:   removing identical section {{.*}}cycle2.o:(.text.f2)

## The same cycle, but f1 has one more instruction.
# RUN: ld.lld cycle-diff.o cycle2.o -o cycle-diff --icf=all --print-icf-sections | count 0

## Chains a and c are identical; b differs only in its last section, farther
## away than the initial hash reaches. Only the eight a/c pairs fold.
# RUN: ld.lld chain.o -o chain --icf=all --print-icf-sections | count 16

#--- self.s
.globl _start
_start:
  ret

.section .text.f1, "ax"
f1:
  call f1

.section .text.f2, "ax"
f2:
  call f2

#--- cycle1.s
.globl _start, f1, f2
_start:
  ret

.section .text.f1, "ax"
f1:
  mov $60, %rdi
  call f2

#--- cycle2.s
.globl f1, f2
.section .text.f2, "ax"
f2:
  mov $60, %rdi
  call f1

#--- cycle-diff.s
.globl _start, f1, f2
_start:
  ret

.section .text.f1, "ax"
f1:
  mov $60, %rdi
  call f2
  mov $0, %rax

#--- chain.s
.macro hop, sym, next
.section .text.\sym, "ax"
\sym: jmp \next
.endm

.macro tail, sym, val
.section .text.\sym, "ax"
\sym: mov \val, %rax
.endm

.globl _start
_start:
  ret

hop a1, a2
hop a2, a3
hop a3, a4
hop a4, a5
hop a5, a6
hop a6, a7
hop a7, a8
tail a8, $1

hop b1, b2
hop b2, b3
hop b3, b4
hop b4, b5
hop b5, b6
hop b6, b7
hop b7, b8
tail b8, $2

hop c1, c2
hop c2, c3
hop c3, c4
hop c4, c5
hop c5, c6
hop c6, c7
hop c7, c8
tail c8, $1
