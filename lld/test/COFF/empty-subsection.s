# REQUIRES: x86
# RUN: llvm-mc -triple=x86_64-windows-msvc -filetype=obj -o %t.obj %s
# RUN: lld-link /entry:main /debug /out:%t.exe %t.obj 2>&1 | FileCheck %s

# CHECK: warning: empty symbols subsection

.globl main
.Lfunc_begin0:
main:
  xorl  %eax, %eax
  retq
.Lfunc_end0:

.section .debug$S,"dr"
	.p2align	2
	.long	4                               # Debug section magic
	.long	241                             # Symbol subsection for globals
  .long .Ltmp5-.Ltmp4                   # Subsection size
.Ltmp4:
.Ltmp5:
