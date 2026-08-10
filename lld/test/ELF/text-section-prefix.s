# REQUIRES: x86
## -z keep-text-section-prefix separates text sections with prefix .text.hot,
## .text.unknown, .text.unlikely, .text.startup, or .text.exit, in the absence
## of a SECTIONS command.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld %t.o -o %t1
# RUN: llvm-readelf -S %t1 | FileCheck --check-prefix=NOKEEP %s
# RUN: ld.lld -z nokeep-text-section-prefix %t.o -o %t2
# RUN: cmp %t1 %t2

# RUN: ld.lld -z keep-text-section-prefix %t.o -o %t.keep
# RUN: llvm-readelf -S %t.keep | FileCheck --check-prefix=KEEP %s

# KEEP:      [ 1] .text
# KEEP-NEXT: [ 2] .text.hot
# KEEP-NEXT: [ 3] .text.unknown
# KEEP-NEXT: [ 4] .text.split
# KEEP-NEXT: [ 5] .text.startup
# KEEP-NEXT: [ 6] .text.exit
# KEEP-NEXT: [ 7] .text.unlikely

# NOKEEP:    [ 1] .text
# NOKEEP-NOT:     .text

## With a SECTIONS command, orphan sections are created verbatim.
## No grouping is performed for them.
# RUN: echo 'SECTIONS {}' > %t.lds
# RUN: ld.lld -T %t.lds -z keep-text-section-prefix %t.o -o %t.script
# RUN: llvm-readelf -S %t.script | FileCheck --check-prefix=SCRIPT %s

# SCRIPT:      .text
# SCRIPT-NEXT: .text.f
# SCRIPT-NEXT: .text.hot.f_hot
# SCRIPT-NEXT: .text.unknown.f_unknown
# SCRIPT-NEXT: .text.split.f_split
# SCRIPT-NEXT: .text.startup.f_startup
# SCRIPT-NEXT: .text.exit.f_exit
# SCRIPT-NEXT: .text.unlikely.f_unlikely

.globl _start
_start:
  ret

.section .text.f,"ax"
  nop

.section .text.hot.f_hot,"ax"
  nop

.section .text.unknown.f_unknown,"ax"
  nop

.section .text.split.f_split,"ax"
  nop

.section .text.startup.f_startup,"ax"
  nop

.section .text.exit.f_exit,"ax"
  nop

.section .text.unlikely.f_unlikely,"ax"
  nop
