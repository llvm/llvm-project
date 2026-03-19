## Check that trampoline functions are excluded from density computation.

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: link_fdata %s %t %t.preagg PREAGG
# RUN: llvm-strip -NLjmp %t
# RUN: perf2bolt %t -p %t.preagg --pa -o %t.fdata | FileCheck %s
# CHECK: Functions with density >= {{.*}} account for 99.00% total sample counts.
# CHECK-NOT: the output profile is empty or the --profile-density-cutoff-hot option is set too low.

  .text
  .globl trampoline
trampoline:
  mov main,%rax
  jmpq *%rax
.size trampoline,.-trampoline
# PREAGG: f #trampoline# #trampoline# 2

	.globl main
main:
	.cfi_startproc
	vmovaps %zmm31,%zmm3

	add    $0x4,%r9
	add    $0x40,%r10
	dec    %r14
Ljmp:
	jne    main
# PREAGG: T #Ljmp# #main# #Ljmp# 10
	ret
	.cfi_endproc
.size main,.-main
