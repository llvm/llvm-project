## Check that ICP recognizes functions folded by ICF and inserts a single check

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: ld.lld -q -o %t %t.o

# Without ICF, ICP should not be performed:
# RUN: llvm-bolt %t -o %t.bolt1 --icp=calls --icp-calls-topn=1 --print-icp \
# RUN:   --icp-calls-total-percent-threshold=90 \
# RUN:   --data %t.fdata | FileCheck %s --check-prefix=CHECK-NO-ICF

# CHECK-NO-ICF: ICP percentage of indirect callsites that are optimized = 0.0%

# With ICF, ICP should be performed:
# RUN: llvm-bolt %t -o %t.bolt1 --icp=calls --icp-calls-topn=1 --print-icp \
# RUN:   --icp-calls-total-percent-threshold=90 \
# RUN:   --data %t.fdata --icf | FileCheck %s --check-prefix=CHECK-ICF

# CHECK-ICF: ICP percentage of indirect callsites that are optimized = 100.0%
# CHECK-ICF: Binary Function "main" after indirect-call-promotion
# CHECK-ICF: callq bar

  .globl bar
bar:
	imull	$0x64, %edi, %eax
	addl	$0x2a, %eax
	retq
.size bar, .-bar

  .globl foo
foo:
	imull	$0x64, %edi, %eax
	addl	$0x2a, %eax
	retq
.size foo, .-foo

  .globl main
main:
  pushq %rax
  movslq  %edi, %rax
  leaq  funcs(%rip), %rcx
  xorl  %edi, %edi
LBB00_br:
  callq *(%rcx,%rax,8)
# FDATA: 1 main #LBB00_br# 1 foo 0 0 2
# FDATA: 1 main #LBB00_br# 1 bar 0 0 2
  popq  %rcx
  retq
.size main, .-main

	.section .rodata
	.globl	funcs
funcs:
	.quad	foo
	.quad	bar
